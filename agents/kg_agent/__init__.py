import logging
from .kg_manager import get_kg_manager
from .cypher_query_llm import CypherQueryService
from .context_filter import ContextFilterEmbedding
from .response_generator import ResponseGenerator
from typing import Dict, List, Any

class KGQueryEngine:
    """
    High-level query interface for Knowledge Graph operations.
    """
    
    def __init__(self, patient_query_engine):
        self.logger = logging.getLogger(__name__)
        self.kg_manager = get_kg_manager()
        self.cypher_service = CypherQueryService()
        self.context_filter = ContextFilterEmbedding()
        self.response_generator = ResponseGenerator(patient_query_engine)
    
    def retrieve_medical_context(self, question: str) -> Dict[str, Any]:
        """Retrieve medical context from knowledge graph"""
        return self.cypher_service.retrieve_context_from_kg(question)
    
    def filter_context_for_patient(self, kg_context: List[Dict], patient_info: Dict, question: str):
        """Filter KG context based on patient profile"""
        return self.context_filter.filter_context(kg_context, patient_info, question)
    
    def generate_medical_response(self, question: str, patient_id: str) -> str:
        """Generate complete medical response using filtered KG context"""
        return self.response_generator.generate_response(question, patient_id)
    
    def get_disease_information(self, disease_name: str) -> List[Dict]:
        """Get comprehensive disease information from KG"""
        return self.cypher_service.get_disease_info(disease_name)

    def get_kg_status(self) -> Dict[str, bool]:
        """Get status of KG components"""
        return {
            'graph_connected': self.kg_manager.graph is not None,
            'llm_ready': self.kg_manager.llm is not None,
            'embedding_ready': self.kg_manager.embedding_model is not None,
            'cypher_chain_ready': self.kg_manager.cypher_chain is not None
        }
    
    def clear_caches(self):
        """Clear all caches in KG services"""
        self.cypher_service.clear_cache()
        self.logger.info("KG caches cleared")