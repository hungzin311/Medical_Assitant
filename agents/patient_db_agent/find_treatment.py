import logging
from typing import List, Dict, Any, Tuple, Optional
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range
import json

class TreatmentFinder:
    """
    Handles treatment-related operations for patient records including:
    - Patient record retrieval for treatment planning
    - Finding similar treatment cases for optimization
    """
    
    def __init__(self, patient_vector_store):
        """
        Initialize with reference to the patient vector store.
        
        Args:
            patient_vector_store: Instance of PatientVectorStore for data access
        """
        self.logger = logging.getLogger(__name__)
        self.client = patient_vector_store.client 
        self.collection_name = patient_vector_store.collection_name
        self.embeddings = patient_vector_store.embedding_model
        
    def retrieve_patient_records(
        self, 
        query: str, 
        patient_id: Optional[str] = None,
        demographic_filters: Optional[Dict[str, Any]] = None,
        clinical_filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Multi-stage patient record retrieval (Problem 1).
        
        Args:
            query: Query text
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
        
        query = self.embeddings.embed_query(query)
        # Execute search
        try:
            # Use similarity_search_with_score for vector-based search
            results_with_scores = self.client.query_points(
                query = query,
                collection_name=self.collection_name,
                query_filter=query_filter, 
                limit=limit,
                with_payload = True,
                using="dense"
            )
            
            # The response is a QueryResponse object with a .points attribute
            results = []
            for point in results_with_scores.points:
                result_obj = type('SearchResult', (), {
                    'id': point.id,
                    'score': point.score,
                    'payload': point.payload
                })()
                results.append(result_obj)

            #Retrieved records list
            retrieved_records = []

            # Second search for same patient records only if patient_id is provided
            if patient_id:
                result2 = self.client.query_points( 
                    collection_name=self.collection_name, 
                    query_filter = Filter(
                        must = [
                            FieldCondition(key="patient_id", match=MatchValue(value=patient_id))
                        ]
                    ), 
                    limit = limit, 
                    with_payload= True,
                    using="dense"
                )
                result2 = result2.points
                #Format results
                for result in result2: 
                    record = {
                        "id": result.id,
                        "score": result.score,
                        "payload": result.payload
                    }
                    retrieved_records.append(record)
                        
            # Format results
            for result in results:
                record = {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                }
                if record['id'] != patient_id:
                    retrieved_records.append(record)
            
            self.logger.info(f"Retrieved {len(retrieved_records)} patient records")
            return retrieved_records
            
        except Exception as e:
            self.logger.error(f"Error retrieving patient records: {e}")
            return []

    def find_treatment_cases(
        self, 
        query: str, 
        candidate_treatments: List[str],
        age_range: Optional[Tuple[int, int]] = None,
        comorbidities: Optional[List[str]] = None,
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Find similar cases for treatment optimization (Problem 2).
        
        Args:
            query: Patient query text
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
        query = self.embeddings.embed_query(query)
        try:
            # Use similarity_search_with_score for vector-based search
            results_with_scores = self.client.query_points(
                query=query,
                collection_name=self.collection_name,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                using="dense"
            )
            
            # Convert to expected format - QueryResponse has .points attribute
            results = []
            for point in results_with_scores.points:
                result_obj = type('SearchResult', (), {
                    'id': point.id,
                    'score': point.score,
                    'payload': point.payload
                })()
                results.append(result_obj)
            
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