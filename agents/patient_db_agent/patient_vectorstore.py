import os
import logging
from uuid import uuid4
from datetime import datetime
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, OptimizersConfigDiff


class PatientVectorStore:
    """
    Core vector store operations for patient medical records including:
    - Patient record ingestion and storage
    - Vector embedding and indexing
    - Collection management and optimization
    
    Note: Specialized operations have been moved to dedicated classes:
    - TreatmentFinder: Patient record retrieval and treatment case finding
    - DiseaseEvaluator: Population monitoring and risk assessment
    """
    
    def __init__(self, config, collection_name):
        self.logger = logging.getLogger(__name__)
        self.collection_name = collection_name
        self.embedding_dim = config.rag.embedding_dim
        self.embedding_model = config.rag.embedding_model
        self.retrieval_top_k = config.rag.top_k
        
        # Cloud configuration
        self.qdrant_url = config.rag.url
        self.qdrant_api_key = config.rag.api_key
        self.llm = config.patient_db.llm
        
        # Initialize cloud client
        if not self.qdrant_url or not self.qdrant_api_key:
            self.logger.error("Qdrant cloud URL or API key not provided. Check your environment variables.")
            raise ValueError("Qdrant cloud URL or API key not provided")
            
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )

        if not self._does_collection_exist():
            self._create_patient_collection()

    def _does_collection_exist(self) -> bool:
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
        if outcome_score >= 0.8:
            return "excellent"
        elif outcome_score >= 0.6:
            return "good"
        elif outcome_score >= 0.4:
            return "fair"
        else:
            return "poor"

    def batch_ingest_patients(self, patient_records: List[Dict[str, Any]]) -> List[str]:

        # Ensure collection exists
        if not self._does_collection_exist():
            self._create_patient_collection()
        
        points = []
        record_ids = []
        
        for patient_data in patient_records:
            # Generate record ID
            record_id = str(uuid4())  # Use UUID format for Qdrant compatibility
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
