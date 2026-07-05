import logging
from typing import Any, Dict, List

from .retrieval_coordinator import KGRetrievalCoordinator

logging.getLogger("httpx").disabled = True

class KGQueryEngine:
    """
    High-level query interface for Knowledge Graph operations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.retrieval_coordinator = KGRetrievalCoordinator()
        self.cypher_service = self.retrieval_coordinator.cypher_query_llm
        self.context_filter = self.retrieval_coordinator.context_filter
    
    def retrieve_medical_context(self, question: str) -> Dict[str, Any]:
        """Retrieve medical context from knowledge graph"""
        return self.cypher_service.retrieve_context_from_kg(question)
    
    def filter_context_for_patient(self, kg_context: List[Dict], patient_info: str, question: str):
        """Filter KG context based on patient profile"""
        return self.context_filter.filter_context(kg_context, patient_info, question)
        
    def clear_caches(self):
        """Clear all caches in KG services"""
        self.cypher_service.clear_cache()
        self.logger.info("KG caches cleared")
