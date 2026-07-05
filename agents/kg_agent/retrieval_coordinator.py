import json
import logging
from typing import Any, Dict, List, Optional

from agents.rag_agent.query_expander import QueryExpander
from utils.config import Config

from .context_filter import ContextFilterEmbedding
from .cypher_query_llm import CypherQueryService


class KGRetrievalCoordinator:
    """Shared KG retrieval utilities used by the LangGraph orchestration layer."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cypher_query_llm = CypherQueryService()
        self.context_filter = ContextFilterEmbedding()
        self.query_expander = QueryExpander(Config())
        self.cached_kg_candidates = None
        self.cached_kg_candidates_by_key = {}

    def _cache_key(self, patient_id: Optional[str] = None) -> str:
        return patient_id or "default"

    def set_cached_kg_candidates(
        self,
        candidates: List[Dict[str, Any]],
        patient_id: Optional[str] = None,
    ) -> None:
        kg_candidates_json = json.dumps(candidates, ensure_ascii=False)
        self.cached_kg_candidates = kg_candidates_json
        self.cached_kg_candidates_by_key[self._cache_key(patient_id)] = kg_candidates_json

    def get_cached_kg_candidates(self, patient_id: Optional[str] = None) -> List[Dict[str, Any]]:
        cached = self.cached_kg_candidates_by_key.get(self._cache_key(patient_id)) or self.cached_kg_candidates
        if not cached:
            return []
        if isinstance(cached, list):
            return cached
        try:
            parsed = json.loads(cached)
        except (TypeError, json.JSONDecodeError):
            self.logger.warning("Cached KG candidates could not be parsed")
            return []
        return parsed if isinstance(parsed, list) else []
