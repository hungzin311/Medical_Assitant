import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


class MedlinePlusRetriever:
    """Standalone retriever for the MedlinePlus KB collection.

    This retriever intentionally reads the VectorDB MedlinePlus payload schema:
    payload.doc_id, payload.doc_type, payload.text, payload.metadata.
    """

    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.collection_name = config.collection_name
        self.data_dir = Path(config.data_dir)
        self.embedding_model = config.embedding_model
        self.top_k = config.top_k
        self.vector_name = config.vector_name
        self.client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        top_k = top_k or self.top_k
        linked_entities = self._link_entities_from_json(query)
        query_vector = [float(value) for value in self.embedding_model.embed_query(query)]

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=self.vector_name,
            limit=top_k * 3,
            with_payload=True,
            with_vectors=False,
        )

        raw_results = [
            {
                "score": float(point.score),
                "payload": point.payload or {},
            }
            for point in response.points
        ]
        ranked_results = self._rerank(raw_results, linked_entities)[:top_k]
        documents = [self._to_document(result) for result in ranked_results]

        return {
            "query": query,
            "linked_entities": linked_entities,
            "documents": documents,
        }

    def _load_json(self, path_name: str) -> Any:
        path = self.data_dir / path_name
        return json.loads(path.read_text(encoding="utf-8"))

    def _aliases_for_entity(self, entity: Dict[str, Any]) -> List[str]:
        title = entity.get("title") or ""
        aliases = [title, entity.get("id")]
        aliases.extend(entity.get("also_called") or [])
        if title.lower().endswith(" test"):
            aliases.append(title[:-5])
        aliases.extend(re.findall(r"\(([^)]+)\)", title))
        return [alias for alias in aliases if alias]

    def _link_entities_from_json(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        query_norm = _normalize(query)
        if not query_norm or not self.data_dir.exists():
            return []

        candidates: List[Dict[str, Any]] = []
        for path_name in ["health_topics.json", "lab_tests.json"]:
            path = self.data_dir / path_name
            if not path.exists():
                continue
            for entity in self._load_json(path_name):
                for alias in self._aliases_for_entity(entity):
                    alias_norm = _normalize(alias.replace("-", " "))
                    if len(alias_norm) >= 3 and alias_norm in query_norm:
                        # only 1 alias per entity
                        candidates.append(
                            {
                                "entity_id": entity["id"],
                                "entity_type": entity["entity_type"],
                                "title": entity["title"],
                                "matched_alias": alias,
                                "_alias_len": len(alias_norm),
                            }
                        )
                        break
        # filter duplicates 
        candidates.sort(key=lambda item: item["_alias_len"], reverse=True)
        linked: List[Dict[str, Any]] = []
        seen = set()
        for item in candidates:
            key = (item["entity_type"], item["entity_id"])
            if key in seen:
                continue
            seen.add(key)
            item.pop("_alias_len", None)
            linked.append(item)
            if len(linked) >= limit:
                break
        return linked

    def _rerank(
        self,
        results: List[Dict[str, Any]],
        linked_entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        linked_ids = {item["entity_id"] for item in linked_entities}

        def rank_score(result: Dict[str, Any]) -> float:
            payload = result.get("payload") or {}
            metadata = payload.get("metadata") or {}
            score = float(result.get("score") or 0.0)
            if metadata.get("entity_id") in linked_ids:
                score += 0.1
            return score

        return sorted(results, key=rank_score, reverse=True)

    def _to_document(self, result: Dict[str, Any]) -> Dict[str, Any]:
        payload = result.get("payload") or {}
        metadata = payload.get("metadata") or {}
        return {
            "id": payload.get("doc_id", "unknown"),
            "content": payload.get("text", ""),
            "score": float(result.get("score") or 0.0),
            "source": "MedlinePlus",
            "source_path": metadata.get("source_url", ""),
            "title": metadata.get("title", ""),
            "doc_type": payload.get("doc_type", ""),
            "entity_id": metadata.get("entity_id", ""),
            "entity_type": metadata.get("entity_type", ""),
            "section_type": metadata.get("section_type", ""),
            "metadata": metadata,
        }
