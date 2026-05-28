import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import AsyncQdrantClient


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
        self.include_relations = config.include_relations
        self.max_relations = config.max_relations
        self.client = AsyncQdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aretrieve(query=query, top_k=top_k))
        raise RuntimeError("retrieve() was called inside an event loop; use await aretrieve() instead")

    async def aretrieve(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        top_k = top_k or self.top_k
        linked_entities = self._link_entities_from_json(query)
        intent = self._infer_intent(query)
        query_vector = [float(value) for value in await self.embedding_model.aembed_query(query)]

        response = await self.client.query_points(
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
        ranked_results = self._rerank(raw_results, intent, linked_entities)[:top_k]
        documents = [self._to_document(result) for result in ranked_results]

        return {
            "query": query,
            "intent": intent,
            "linked_entities": linked_entities,
            "expanded_relations": self._expand_relations(linked_entities) if self.include_relations else [],
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

    def _infer_intent(self, query: str) -> str:
        query_l = _normalize(query)
        if any(token in query_l for token in ["mean", "meaning", "result", "cao", "thấp", "ket qua", "kết quả"]):
            return "interpretation"
        if any(token in query_l for token in ["test", "xét nghiệm", "xet nghiem", "cần xét nghiệm", "can xet nghiem"]):
            return "lab_test_lookup"
        return "general"

    def _rerank(
        self,
        results: List[Dict[str, Any]],
        intent: str,
        linked_entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        linked_ids = {item["entity_id"] for item in linked_entities}

        def rank_score(result: Dict[str, Any]) -> float:
            payload = result.get("payload") or {}
            metadata = payload.get("metadata") or {}
            score = float(result.get("score") or 0.0)
            if metadata.get("entity_id") in linked_ids:
                score += 0.2
            if intent == "interpretation" and metadata.get("section_type") == "interpretation":
                score += 0.25
            if intent == "lab_test_lookup" and payload.get("doc_type") in {"relation_summary", "entity_profile"}:
                score += 0.15
            return score

        return sorted(results, key=rank_score, reverse=True)

    def _expand_relations(self, linked_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not linked_entities or not self.data_dir.exists():
            return []

        path = self.data_dir / "relations.json"
        if not path.exists():
            return []

        linked_keys = {(item["entity_type"], item["entity_id"]) for item in linked_entities}
        expanded = []
        for relation in self._load_json("relations.json"):
            source_key = (relation["source_type"], relation["source_id"])
            if source_key in linked_keys:
                expanded.append(relation)
            if len(expanded) >= self.max_relations:
                break
        return expanded

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
