import logging
from .cypher_query_llm import CypherQueryService
from .context_filter import ContextFilterEmbedding
from .response_generator import ResponseGenerator
from typing import Dict, List, Any, Optional
from agents.patient_db_agent import PatientQueryEngine

class KGQueryEngine:
    """
    High-level query interface for Knowledge Graph operations.
    """
    
    def __init__(self, patient_query_engine: PatientQueryEngine):
        self.logger = logging.getLogger(__name__)
        self.cypher_service = CypherQueryService()
        self.context_filter = ContextFilterEmbedding()
        self.response_generator = ResponseGenerator(patient_query_engine)
    
    def retrieve_medical_context(self, question: str) -> Dict[str, Any]:
        """Retrieve medical context from knowledge graph"""
        return self.cypher_service.retrieve_context_from_kg(question)
    
    def filter_context_for_patient(self, kg_context: List[Dict], patient_info: Dict, question: str):
        """Filter KG context based on patient profile"""
        return self.context_filter.filter_context(kg_context, patient_info, question)
    
    def generate_medical_response(self, question: str, patient_id: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate complete medical response using filtered KG context"""
        return self.response_generator.generate_response(question, patient_id, chat_history)
    
    def get_disease_information(self, disease_name: str) -> List[Dict]:
        """Get comprehensive disease information from KG"""
        return self.cypher_service.get_disease_info(disease_name)
    
    def clear_caches(self):
        """Clear all caches in KG services"""
        self.cypher_service.clear_cache()
        self.logger.info("KG caches cleared")