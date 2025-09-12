from .kg_manager import get_kg_manager
from typing import Dict, List, Any, Optional
from prompt import cypher_query

class CypherQueryService: 
    def __init__(self):
        self.kg_manager = get_kg_manager()
        self.cypher_query = cypher_query
        
    def retrieve_context_from_kg(self, question: str) -> Dict[str, Any]:
        full_result = self.kg_manager.cypher_chain.invoke(question)
        return full_result['result']
    
    def execute_cypher_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        return self.kg_manager.graph.query(query, params=params or {})

    def get_disease_info(self, disease_name: str) -> List[Dict]:
        query = cypher_query
        return self.execute_cypher_query(query, params={"disease_name": disease_name})
