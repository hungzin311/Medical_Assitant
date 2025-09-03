import logging
from typing import List, Dict, Any, Tuple, Optional
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range


class TreatmentFinder:

    def __init__(self, patient_vector_store):
        self.logger = logging.getLogger(__name__)
        self.client = patient_vector_store.client 
        self.collection_name = patient_vector_store.collection_name
        self.embeddings = patient_vector_store.embedding_model
        self.llm = patient_vector_store.llm
    def find_treatment_cases(
        self, 
        query: str, 
        patient_id: str,
        candidate_treatments: List[str] = None,
        age_range: Optional[Tuple[int, int]] = None,
        comorbidities: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:

        must_conditions = [ 
            FieldCondition(key="primary_outcome", range=Range(gte=0.0))
        ]
        
        must_not_conditions = [ 
            FieldCondition(key="patient_id", match=MatchValue(value=patient_id))
        ]  

        should_conditions = []
        
        # Treatment filters
        if candidate_treatments:
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
        
        query_filter = Filter(must_not = must_not_conditions, must=must_conditions, should=should_conditions)
    
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
                    "patient_id": result.payload.get("patient_id", ""),
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
    def recommend_treatment_llm(
            self, 
            patient_records: List[Dict[str, Any]],
            treatment_references: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        
        #Simple version. Does not contain RAG or any technique. Like a demo, about 50% of the full version.
    
        prompt = f"""
        NHIỆM VỤ: ĐỀ XUẤT ĐIỀU TRỊ

        Hãy đề xuất điều trị dựa trên các trường hợp tương tự đã có.

        {patient_records}

        {treatment_references}

        """
        response = self.llm.invoke(prompt)
        return response.content

       