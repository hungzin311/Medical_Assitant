import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from .retriever import MedlinePlusRetriever

class MedlinePlusAgent:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config.medlineplus
        self.retriever = MedlinePlusRetriever(self.config)
        self.llm = self.config.llm

    def _extract_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        sources = []
        seen = set()
        for doc in documents:
            key = (doc.get("title"), doc.get("source_path"))
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "title": doc.get("title", "MedlinePlus"),
                    "source": doc.get("source", "MedlinePlus"),
                    "source_path": doc.get("source_path", ""),
                }
            )
        return sources

    def _estimate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        if not documents:
            return 0.0
        top_score = max(float(doc.get("score") or 0.0) for doc in documents)
        return max(0.0, min(1.0, top_score))
