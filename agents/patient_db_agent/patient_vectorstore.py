import os
import re
import logging
import json
from uuid import uuid4
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, OptimizersConfigDiff, Filter, FieldCondition, MatchValue, Range


class PatientVectorStore:
    """
    Specialized vector store for patient medical records supporting:
    - Problem 1: Multi-dimensional patient record retrieval
    - Problem 2: Treatment optimization with outcome tracking
    - Problem 3: Early warning system and population health monitoring
    """
    
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.collection_name = "medical_records"
        self.embedding_dim = config.rag.embedding_dim
        self.embedding_model = config.rag.embedding_model
        self.retrieval_top_k = config.rag.top_k
        
        # Cloud configuration
        self.qdrant_url = config.rag.url
        self.qdrant_api_key = config.rag.api_key
        
        # Initialize cloud client
        if not self.qdrant_url or not self.qdrant_api_key:
            self.logger.error("Qdrant cloud URL or API key not provided. Check your environment variables.")
            raise ValueError("Qdrant cloud URL or API key not provided")
            
        self.logger.info(f"Connecting to Qdrant cloud for patient database at {self.qdrant_url}")
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )

    def _does_collection_exist(self) -> bool:
        """Check if the patient collection already exists in Qdrant cloud."""
        try:
            collection_info = self.client.get_collections()
            collection_names = [collection.name for collection in collection_info.collections]
            return self.collection_name in collection_names
        except Exception as e:
            self.logger.error(f"Error checking for collection existence: {e}")
            return False

    def _create_patient_collection(self):
        """Create a new patient collection with optimized indexing."""
        try:
            # Delete collection if it exists
            if self._does_collection_exist():
                self.logger.info(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(collection_name=self.collection_name)
                
            # Create collection with vector configuration
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": VectorParams(size=self.embedding_dim, distance=Distance.COSINE)},
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=0,  # Build index immediately
                )
            )
            
            # Create payload indexes for efficient filtering
            payload_indexes = [
                # Patient identification
                ("patient_id", "keyword"),
                ("visit_id", "keyword"),
                ("record_type", "keyword"),
                
                # Temporal filters
                ("timestamp", "datetime"),
                ("visit_date", "keyword"),
                ("is_recent", "bool"),
                
                # Demographics (Problem 1)
                ("age_group", "keyword"),
                ("sex", "keyword"),
                ("bmi_category", "keyword"),
                
                # Clinical filters
                ("severity_category", "keyword"),
                ("comorbidities", "keyword"),
                ("clinical_tags", "keyword"),
                
                # Treatment filters (Problem 2)
                ("candidate_treatments", "keyword"),
                ("outcome_category", "keyword"),
                ("contraindications", "keyword"),
                
                # Population health (Problem 3)
                ("facility_id", "keyword"),
                ("geographic_region", "keyword"),
                ("outbreak_risk", "bool"),
                ("high_risk_patient", "bool"),
                ("treatment_resistance", "bool"),
                ("unusual_presentation", "bool"),
                
                # Range filters
                ("age", "integer"),
                ("severity_score", "float"),
                ("primary_outcome", "float"),
                ("comorbidity_count", "integer"),
            ]
            
            # Create indexes
            for field_name, field_type in payload_indexes:
                try:
                    if field_type == "keyword":
                        self.client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name=field_name,
                            field_schema="keyword"
                        )
                    elif field_type == "integer":
                        self.client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name=field_name,
                            field_schema="integer"
                        )
                    elif field_type == "float":
                        self.client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name=field_name,
                            field_schema="float"
                        )
                    elif field_type == "bool":
                        self.client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name=field_name,
                            field_schema="bool"
                        )
                    elif field_type == "datetime":
                        self.client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name=field_name,
                            field_schema="datetime"
                        )
                except Exception as idx_error:
                    self.logger.warning(f"Could not create index for {field_name}: {idx_error}")
                    
            self.logger.info(f"Created new patient collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error creating patient collection: {e}")
            raise e

    def ingest_patient_record(self, patient_data: Dict[str, Any]) -> str:
        """
        Ingest a single patient record into the vector store.
        
        Args:
            patient_data: Dictionary containing patient information
            
        Returns:
            Record ID of the inserted document
        """
        # Ensure collection exists
        if not self._does_collection_exist():
            self._create_patient_collection()
            
        # Generate record ID
        record_id = f"{patient_data['patient_id']}_{patient_data['visit_date']}_{uuid4().hex[:6]}"
        
        # Create embedding from clinical text
        text_content = f"{patient_data.get('summary_text', '')} {patient_data.get('chief_complaint', '')}"
        vector = self.embedding_model.embed_query(text_content)
        
        # Build payload
        payload = self._build_patient_payload(patient_data)
        
        try:
            # Insert into Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[{
                    "id": record_id,
                    "vector": {"dense": vector},
                    "payload": payload
                }]
            )
            
            self.logger.info(f"Successfully ingested patient record: {record_id}")
            return record_id
            
        except Exception as e:
            self.logger.error(f"Error ingesting patient record: {e}")
            raise e

    def _build_patient_payload(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build standardized payload from patient data."""
        
        # Set defaults and process data
        current_time = datetime.now().isoformat()
        visit_date = patient_data.get('visit_date', datetime.now().strftime('%Y-%m-%d'))
        
        # Calculate age group
        age = patient_data.get('age', 0)
        if age < 18:
            age_group = "0-18"
        elif age < 30:
            age_group = "18-30"
        elif age < 50:
            age_group = "30-50"
        elif age < 65:
            age_group = "50-65"
        else:
            age_group = "65+"
            
        # Calculate BMI category
        bmi = patient_data.get('bmi', 0)
        if bmi < 18.5:
            bmi_category = "underweight"
        elif bmi < 25:
            bmi_category = "normal"
        elif bmi < 30:
            bmi_category = "overweight"
        else:
            bmi_category = "obese"
            
        # Determine if recent (last 90 days)
        try:
            visit_dt = datetime.strptime(visit_date, '%Y-%m-%d')
            is_recent = (datetime.now() - visit_dt).days <= 90
        except:
            is_recent = True
            
        # Build comprehensive payload
        payload = {
            # Patient identification
            "patient_id": patient_data.get('patient_id', ''),
            "visit_id": patient_data.get('visit_id', ''),
            "record_type": patient_data.get('record_type', 'encounter'),
            
            # Temporal context
            "timestamp": patient_data.get('timestamp', current_time),
            "visit_date": visit_date,
            "disease_duration_days": patient_data.get('disease_duration_days', 0),
            "days_since_last_visit": patient_data.get('days_since_last_visit', 0),
            "is_recent": is_recent,
            
            # Clinical text
            "summary_text": patient_data.get('summary_text', ''),
            "chief_complaint": patient_data.get('chief_complaint', ''),
            
            # Demographics
            "age": age,
            "age_group": age_group,
            "sex": patient_data.get('sex', ''),
            "bmi": bmi,
            "bmi_category": bmi_category,
            
            # Clinical features
            "vital_signs": patient_data.get('vital_signs', {}),
            "lab_values": patient_data.get('lab_values', {}),
            "severity_score": patient_data.get('severity_score', 0.0),
            "severity_category": patient_data.get('severity_category', 'mild'),
            
            # Treatment data
            "treatments_tried": patient_data.get('treatments_tried', []),
            "candidate_treatments": patient_data.get('candidate_treatments', []),
            "contraindications": patient_data.get('contraindications', []),
            
            # Comorbidities
            "comorbidities": patient_data.get('comorbidities', []),
            "comorbidity_count": len(patient_data.get('comorbidities', [])),
            "risk_factors": patient_data.get('risk_factors', []),
            
            # Outcomes
            "primary_outcome": patient_data.get('primary_outcome', 0.0),
            "readmission_30d": patient_data.get('readmission_30d', False),
            "adverse_events": patient_data.get('adverse_events', []),
            "disease_progression": patient_data.get('disease_progression', 'stable'),
            
            # Population health
            "facility_id": patient_data.get('facility_id', ''),
            "provider_id": patient_data.get('provider_id', ''),
            "geographic_region": patient_data.get('geographic_region', ''),
            "insurance_type": patient_data.get('insurance_type', ''),
            
            # Pattern flags
            "outbreak_risk": patient_data.get('outbreak_risk', False),
            "unusual_presentation": patient_data.get('unusual_presentation', False),
            "treatment_resistance": patient_data.get('treatment_resistance', False),
            "quality_alert": patient_data.get('quality_alert', False),
            "high_risk_patient": patient_data.get('high_risk_patient', False),
            
            # Search optimization
            "clinical_tags": patient_data.get('clinical_tags', []),
            "embedding_type": "clinical_summary",
            "data_completeness": patient_data.get('data_completeness', 1.0),
            
            # Research
            "research_cohort": patient_data.get('research_cohort', []),
            "evidence_level": patient_data.get('evidence_level', 'medium'),
            "followup_available": patient_data.get('followup_available', False),
            
            # Outcome categories for filtering
            "outcome_category": self._categorize_outcome(patient_data.get('primary_outcome', 0.0))
        }
        
        return payload

    def _categorize_outcome(self, outcome_score: float) -> str:
        """Categorize outcome score for filtering."""
        if outcome_score >= 0.8:
            return "excellent"
        elif outcome_score >= 0.6:
            return "good"
        elif outcome_score >= 0.4:
            return "fair"
        else:
            return "poor"

    # Problem 1: Patient Record Retrieval Methods
    def retrieve_patient_records(
        self, 
        query_vector: List[float], 
        patient_id: Optional[str] = None,
        demographic_filters: Optional[Dict[str, Any]] = None,
        clinical_filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Multi-stage patient record retrieval (Problem 1).
        
        Args:
            query_vector: Query embedding vector
            patient_id: Optional patient ID for same-patient history
            demographic_filters: Age, sex, BMI filters
            clinical_filters: Comorbidities, severity, diagnosis filters
            limit: Maximum number of results
            
        Returns:
            List of retrieved patient records with scores
        """
        must_conditions = []
        should_conditions = []
        
        # Stage 1: Same patient history (highest priority)
        if patient_id:
            should_conditions.append(
                FieldCondition(key="patient_id", match=MatchValue(value=patient_id))
            )
        
        # Stage 2: Demographic filters
        if demographic_filters:
            if "age_range" in demographic_filters:
                min_age, max_age = demographic_filters["age_range"]
                must_conditions.append(
                    FieldCondition(key="age", range=Range(gte=min_age, lte=max_age))
                )
            if "sex" in demographic_filters:
                should_conditions.append(
                    FieldCondition(key="sex", match=MatchValue(value=demographic_filters["sex"]))
                )
            if "bmi_category" in demographic_filters:
                should_conditions.append(
                    FieldCondition(key="bmi_category", match=MatchValue(value=demographic_filters["bmi_category"]))
                )
        
        # Stage 3: Clinical filters
        if clinical_filters:
            if "comorbidities" in clinical_filters:
                for comorbidity in clinical_filters["comorbidities"]:
                    should_conditions.append(
                        FieldCondition(key="comorbidities", match=MatchValue(value=comorbidity))
                    )
            if "severity_category" in clinical_filters:
                must_conditions.append(
                    FieldCondition(key="severity_category", match=MatchValue(value=clinical_filters["severity_category"]))
                )
        
        # Build filter
        query_filter = None
        if must_conditions or should_conditions:
            query_filter = Filter(
                must=must_conditions if must_conditions else None,
                should=should_conditions if should_conditions else None
            )
        
        # Execute search
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("dense", query_vector),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            result2 = self.client.search( 
                collection_name=self.collection_name, 
                query_filter = Filter(
                    must = [
                        FieldCondition(key="patient_id", match=MatchValue(value=patient_id))
                    ]
                ), 
                limit = limit, 
                with_payload= True
            )
            
            # Format results
            retrieved_records = []
            for result in results:
                record = {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                }
                if record['id'] != patient_id:
                    retrieved_records.append(record)
            
            for result in result2: 
                record = {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                }
                retrieved_records.append(record)
                
            self.logger.info(f"Retrieved {len(retrieved_records)} patient records")
            return retrieved_records
            
        except Exception as e:
            self.logger.error(f"Error retrieving patient records: {e}")
            return []

    # Problem 2: Treatment Optimization Methods
    def find_treatment_cases(
        self, 
        query_vector: List[float], 
        candidate_treatments: List[str],
        age_range: Optional[Tuple[int, int]] = None,
        comorbidities: Optional[List[str]] = None,
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Find similar cases for treatment optimization (Problem 2).
        
        Args:
            query_vector: Patient embedding vector
            candidate_treatments: List of candidate treatments
            age_range: Optional age range filter
            comorbidities: Optional comorbidity filters
            limit: Maximum number of results
            
        Returns:
            List of similar cases with treatment outcomes
        """
        must_conditions = [
            # Must have outcome data
            FieldCondition(key="primary_outcome", range=Range(gte=0.0))
        ]
        
        should_conditions = []
        
        # Treatment filters
        for treatment in candidate_treatments:
            should_conditions.append(
                FieldCondition(key="candidate_treatments", match=MatchValue(value=treatment))
            )
        
        # Age range filter
        if age_range:
            min_age, max_age = age_range
            must_conditions.append(
                FieldCondition(key="age", range=Range(gte=min_age, lte=max_age))
            )
        
        # Comorbidity filters
        if comorbidities:
            for comorbidity in comorbidities:
                should_conditions.append(
                    FieldCondition(key="comorbidities", match=MatchValue(value=comorbidity))
                )
        
        query_filter = Filter(must=must_conditions, should=should_conditions)
        
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("dense", query_vector),
                query_filter=query_filter,
                limit=limit,
                with_payload=["treatments_tried", "primary_outcome", "comorbidities", 
                            "age", "sex", "severity_score", "contraindications"],
                with_vectors=False
            )
            
            treatment_cases = []
            for result in results:
                case = {
                    "id": result.id,
                    "distance": 1 - result.score,  # Convert similarity to distance
                    "treatments_tried": result.payload.get("treatments_tried", []),
                    "primary_outcome": result.payload.get("primary_outcome", 0.0),
                    "age": result.payload.get("age", 0),
                    "sex": result.payload.get("sex", ""),
                    "comorbidities": result.payload.get("comorbidities", []),
                    "severity_score": result.payload.get("severity_score", 0.0),
                    "contraindications": result.payload.get("contraindications", [])
                }
                treatment_cases.append(case)
                
            self.logger.info(f"Found {len(treatment_cases)} treatment cases")
            return treatment_cases
            
        except Exception as e:
            self.logger.error(f"Error finding treatment cases: {e}")
            return []

    # Problem 3: Early Warning System Methods
    def monitor_population_patterns(
        self, 
        time_window: str = "7d",
        geographic_region: Optional[str] = None,
        facility_id: Optional[str] = None,
        risk_flags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Monitor population patterns for early warning (Problem 3).
        
        Args:
            time_window: Time window (e.g., "7d", "30d")
            geographic_region: Optional geographic filter
            facility_id: Optional facility filter
            risk_flags: List of risk flags to monitor
            limit: Maximum number of results
            
        Returns:
            List of flagged cases
        """
        must_conditions = []
        should_conditions = []
        
        # Time filter - for simplicity, using is_recent flag
        if time_window in ["7d", "30d", "90d"]:
            must_conditions.append(
                FieldCondition(key="is_recent", match=MatchValue(value=True))
            )
        
        # Geographic filter
        if geographic_region:
            must_conditions.append(
                FieldCondition(key="geographic_region", match=MatchValue(value=geographic_region))
            )
        
        # Facility filter
        if facility_id:
            must_conditions.append(
                FieldCondition(key="facility_id", match=MatchValue(value=facility_id))
            )
        
        # Risk flags
        risk_flag_mapping = {
            "outbreak": "outbreak_risk",
            "unusual": "unusual_presentation",
            "resistance": "treatment_resistance",
            "quality": "quality_alert",
            "high_risk": "high_risk_patient"
        }
        
        if risk_flags:
            for flag in risk_flags:
                if flag in risk_flag_mapping:
                    should_conditions.append(
                        FieldCondition(key=risk_flag_mapping[flag], match=MatchValue(value=True))
                    )
        else:
            # Default: monitor all risk flags
            for flag_field in risk_flag_mapping.values():
                should_conditions.append(
                    FieldCondition(key=flag_field, match=MatchValue(value=True))
                )
        
        query_filter = Filter(
            must=must_conditions if must_conditions else None,
            should=should_conditions if should_conditions else None
        )
        
        try:
            # Use scroll for large result sets
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            flagged_cases = []
            for result in results[0]:  # results is tuple (points, next_page_offset)
                case = {
                    "id": result.id,
                    "payload": result.payload,
                    "risk_score": self._calculate_risk_score(result.payload)
                }
                flagged_cases.append(case)
            
            # Sort by risk score
            flagged_cases.sort(key=lambda x: x["risk_score"], reverse=True)
            
            self.logger.info(f"Found {len(flagged_cases)} flagged cases for monitoring")
            return flagged_cases
            
        except Exception as e:
            self.logger.error(f"Error monitoring population patterns: {e}")
            return []

    def _calculate_risk_score(self, payload: Dict[str, Any]) -> float:
        """Calculate composite risk score for population monitoring."""
        risk_score = 0.0
        
        # Risk flags
        if payload.get("outbreak_risk", False):
            risk_score += 0.3
        if payload.get("unusual_presentation", False):
            risk_score += 0.25
        if payload.get("treatment_resistance", False):
            risk_score += 0.2
        if payload.get("quality_alert", False):
            risk_score += 0.15
        if payload.get("high_risk_patient", False):
            risk_score += 0.1
        
        # Severity contribution
        severity_score = payload.get("severity_score", 0.0)
        risk_score += severity_score * 0.2
        
        # Comorbidity contribution
        comorbidity_count = payload.get("comorbidity_count", 0)
        risk_score += min(comorbidity_count * 0.05, 0.2)
        
        return min(risk_score, 1.0)


    def batch_ingest_patients(self, patient_records: List[Dict[str, Any]]) -> List[str]:
        """
        Batch ingest multiple patient records.
        
        Args:
            patient_records: List of patient data dictionaries
            
        Returns:
            List of record IDs
        """
        # Ensure collection exists
        if not self._does_collection_exist():
            self._create_patient_collection()
        
        points = []
        record_ids = []
        
        for patient_data in patient_records:
            # Generate record ID
            record_id = f"{patient_data['patient_id']}_{patient_data['visit_date']}_{uuid4().hex[:6]}"
            record_ids.append(record_id)
            
            # Create embedding
            text_content = f"{patient_data.get('summary_text', '')} {patient_data.get('chief_complaint', '')}"
            vector = self.embedding_model.embed_query(text_content)
            
            # Build payload
            payload = self._build_patient_payload(patient_data)
            
            points.append({
                "id": record_id,
                "vector": {"dense": vector},
                "payload": payload
            })
        
        try:
            # Batch insert
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            self.logger.info(f"Successfully batch ingested {len(points)} patient records")
            return record_ids
            
        except Exception as e:
            self.logger.error(f"Error batch ingesting patient records: {e}")
            raise e
