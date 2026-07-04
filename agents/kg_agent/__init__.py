import logging
from .cypher_query_llm import CypherQueryService
from .context_filter import ContextFilterEmbedding
from .response_generator import ResponseGenerator
from typing import Dict, List, Any, Optional
logging.getLogger("httpx").disabled = True

class KGQueryEngine:
    """
    High-level query interface for Knowledge Graph operations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cypher_service = CypherQueryService()
        self.context_filter = ContextFilterEmbedding()
        self.response_generator = ResponseGenerator()
    
    def retrieve_medical_context(self, question: str) -> Dict[str, Any]:
        """Retrieve medical context from knowledge graph"""
        return self.cypher_service.retrieve_context_from_kg(question)
    
    def filter_context_for_patient(self, kg_context: List[Dict], patient_info: Dict, question: str):
        """Filter KG context based on patient profile"""
        return self.context_filter.filter_context(kg_context, patient_info, question)
        
    def evaluate_mcq(self, question: str, choices: List[str]) -> Dict[str, Any]:
        return self.response_generator.evaluate_mcq(question, choices)
    
    def get_disease_information(self, disease_name: str) -> List[Dict]:
        """Get comprehensive disease information from KG"""
        return self.cypher_service.get_disease_info(disease_name)
    
    def clear_caches(self):
        """Clear all caches in KG services"""
        self.cypher_service.clear_cache()
        self.logger.info("KG caches cleared")