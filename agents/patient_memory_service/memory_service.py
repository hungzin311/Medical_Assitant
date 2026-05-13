import os
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from .config import PatientMemorySettings, get_patient_memory_settings
from .schemas import (
    PatientConditionCreate,
    PatientConversationMemoryCreate,
    PatientMemoryItem,
    PatientMemoryListRequest,
    PatientMemorySearchRequest,
    utc_now,
)


class PatientMemoryServiceError(RuntimeError):
    """Raised when the patient memory service cannot complete an operation."""


def _import_mem0_memory():
    """Import the installed Mem0 library."""
    try:
        from mem0 import Memory
    except ModuleNotFoundError as exc:
        raise PatientMemoryServiceError(
            "Mem0 is not installed. Install it with `pip install \"mem0ai[nlp]\"` "
            "before running the patient memory service."
        ) from exc
    return Memory


class PatientMemoryService:
    """Mem0 wrapper for durable patient condition memory."""

    def __init__(self, settings: Optional[PatientMemorySettings] = None):
        self.settings = settings or get_patient_memory_settings()
        self._memory = None
        self._lock = Lock()

    @property
    def memory(self):
        if self._memory is None:
            with self._lock:
                if self._memory is None:
                    self._memory = self._build_memory()
        return self._memory

    def _build_memory(self):
        os.environ.setdefault("MEM0_TELEMETRY", "False")
        self._configure_runtime()

        Memory = _import_mem0_memory()
        return Memory.from_config(self._build_mem0_config())

    def _configure_runtime(self) -> None:
        if self.settings.use_project_proxy:
            from utils.proxy_setting import set_proxy

            set_proxy()

        Path(self.settings.history_db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.settings.local_qdrant_path).mkdir(parents=True, exist_ok=True)

    def _build_mem0_config(self) -> Dict[str, Any]:
        from utils.llm_config import get_fpt_vietnamese_embedding, get_gemini_llm

        return {
            "vector_store": {
                "provider": "qdrant",
                "config": self._build_vector_store_config(),
            },
            "llm": {
                "provider": "langchain",
                "config": {
                    "model": get_gemini_llm(temperature=self.settings.llm_temperature),
                    "temperature": self.settings.llm_temperature,
                    "max_tokens": self.settings.llm_max_tokens,
                },
            },
            "embedder": {
                "provider": "langchain",
                "config": {
                    "model": get_fpt_vietnamese_embedding(),
                    "embedding_dims": self.settings.embedding_dims,
                },
            },
            "history_db_path": self.settings.history_db_path,
            "custom_instructions": self.settings.custom_instructions,
        }

    def _build_vector_store_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "collection_name": self.settings.collection_name,
            "embedding_model_dims": self.settings.embedding_dims,
            "on_disk": True,
        }
        if self.settings.qdrant_url and self.settings.qdrant_api_key:
            config.update(
                {
                    "url": self.settings.qdrant_url,
                    "api_key": self.settings.qdrant_api_key,
                    "path": None,
                }
            )
        else:
            config["path"] = self.settings.local_qdrant_path
        return config

    def add_condition(self, payload: PatientConditionCreate) -> Dict[str, Any]:
        metadata = self._condition_metadata(payload)
        memory_text = self._condition_to_memory_text(payload, metadata)
        try:
            result = self.memory.add(
                [{"role": "user", "content": memory_text}],
                user_id=payload.patient_id,
                agent_id=self.settings.agent_id,
                run_id=payload.run_id,
                metadata=metadata,
                infer=False,
            )
        except Exception as exc:
            raise PatientMemoryServiceError(f"Failed to add patient condition memory: {exc}") from exc

        return {
            "status": "success",
            "patient_id": payload.patient_id,
            "run_id": payload.run_id,
            "results": self._normalize_results(result),
        }

    def add_conversation(self, payload: PatientConversationMemoryCreate) -> Dict[str, Any]:
        metadata = deepcopy(payload.metadata)
        metadata.update(
            {
                "memory_kind": "patient_condition_conversation",
                "source": "conversation",
                "stored_at": utc_now().isoformat(),
            }
        )
        messages = [message.model_dump(exclude_none=True) for message in payload.messages]
        try:
            result = self.memory.add(
                messages,
                user_id=payload.patient_id,
                agent_id=self.settings.agent_id,
                run_id=payload.run_id,
                metadata=metadata,
                infer=payload.infer,
            )
        except Exception as exc:
            raise PatientMemoryServiceError(f"Failed to add conversation memory: {exc}") from exc

        return {
            "status": "success",
            "patient_id": payload.patient_id,
            "run_id": payload.run_id,
            "results": self._normalize_results(result),
        }

    def search(self, payload: PatientMemorySearchRequest) -> Dict[str, Any]:
        filters = self._build_filters(
            patient_id=payload.patient_id,
            run_id=payload.run_id,
            condition_types=[item.value if hasattr(item, "value") else item for item in payload.condition_types or []],
            status=payload.status.value if hasattr(payload.status, "value") else payload.status,
            metadata_filter=payload.metadata_filter,
        )
        try:
            result = self.memory.search(
                payload.query,
                filters=filters,
                top_k=payload.top_k,
                threshold=payload.threshold,
            )
        except Exception as exc:
            raise PatientMemoryServiceError(f"Failed to search patient memories: {exc}") from exc

        return {
            "status": "success",
            "patient_id": payload.patient_id,
            "query": payload.query,
            "results": self._normalize_results(result),
        }

    def list_conditions(self, payload: PatientMemoryListRequest) -> Dict[str, Any]:
        filters = self._build_filters(
            patient_id=payload.patient_id,
            run_id=payload.run_id,
            condition_types=[item.value if hasattr(item, "value") else item for item in payload.condition_types or []],
            status=payload.status.value if hasattr(payload.status, "value") else payload.status,
            metadata_filter=payload.metadata_filter,
        )
        try:
            result = self.memory.get_all(filters=filters, top_k=payload.top_k)
        except Exception as exc:
            raise PatientMemoryServiceError(f"Failed to list patient memories: {exc}") from exc

        return {
            "status": "success",
            "patient_id": payload.patient_id,
            "results": self._normalize_results(result),
        }

    def delete_memory(self, memory_id: str) -> Dict[str, str]:
        try:
            result = self.memory.delete(memory_id)
        except Exception as exc:
            raise PatientMemoryServiceError(f"Failed to delete memory {memory_id}: {exc}") from exc
        return {"status": "success", "message": result.get("message", "Memory deleted successfully")}

    def delete_patient_memories(self, patient_id: str, run_id: Optional[str] = None) -> Dict[str, str]:
        try:
            result = self.memory.delete_all(
                user_id=patient_id,
                agent_id=self.settings.agent_id,
                run_id=run_id,
            )
        except Exception as exc:
            raise PatientMemoryServiceError(f"Failed to delete patient memories: {exc}") from exc
        return {"status": "success", "message": result.get("message", "Memories deleted successfully")}

    def health(self) -> Dict[str, Any]:
        vector_mode = "qdrant_cloud" if self.settings.qdrant_url and self.settings.qdrant_api_key else "qdrant_local"
        return {
            "status": "ok",
            "collection_name": self.settings.collection_name,
            "agent_id": self.settings.agent_id,
            "embedding_dims": self.settings.embedding_dims,
            "vector_mode": vector_mode,
        }

    def _condition_metadata(self, payload: PatientConditionCreate) -> Dict[str, Any]:
        observed_at = payload.observed_at or utc_now()
        metadata = {
            "memory_kind": "patient_condition",
            "condition_type": payload.condition_type,
            "status": payload.status,
            "source": payload.source,
            "observed_at": observed_at.isoformat(),
            "stored_at": utc_now().isoformat(),
            "primary_disease": payload.primary_disease or [],
            "severity": payload.severity,
            "confidence": payload.confidence,
            "tags": payload.tags,
        }
        metadata.update(deepcopy(payload.metadata))
        return {key: value for key, value in metadata.items() if value is not None}

    def _condition_to_memory_text(self, payload: PatientConditionCreate, metadata: Dict[str, Any]) -> str:
        parts = [
            f"Patient condition ({metadata.get('condition_type', 'general')})",
            f"status={metadata.get('status', 'unknown')}",
            f"source={metadata.get('source', 'unknown')}",
            f"observed_at={metadata.get('observed_at')}",
        ]
        if payload.severity is not None:
            parts.append(f"severity={payload.severity}/10")
        if payload.primary_disease:
            parts.append(f"related_diseases={', '.join(payload.primary_disease)}")
        parts.append(f"content={payload.condition_text}")
        return "; ".join(parts)

    def _build_filters(
        self,
        patient_id: str,
        run_id: Optional[str] = None,
        condition_types: Optional[List[str]] = None,
        status: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        filters: Dict[str, Any] = {
            "user_id": patient_id,
            "agent_id": self.settings.agent_id,
        }
        if run_id:
            filters["run_id"] = run_id
        if condition_types:
            filters["condition_type"] = {"in": condition_types}
        if status:
            filters["status"] = status
        if metadata_filter:
            filters.update(deepcopy(metadata_filter))
        return filters

    def _normalize_results(self, response: Any) -> List[PatientMemoryItem]:
        normalized: List[PatientMemoryItem] = []
        for item in self._extract_result_items(response):
            if not isinstance(item, dict):
                continue
            item_data = {
                "id": item.get("id", ""),
                "memory": item.get("memory") or item.get("text") or "",
                "score": item.get("score"),
                "event": item.get("event"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "user_id": item.get("user_id"),
                "agent_id": item.get("agent_id"),
                "run_id": item.get("run_id"),
                "metadata": item.get("metadata") or {},
            }
            if item_data["id"] and item_data["memory"]:
                normalized.append(PatientMemoryItem(**item_data))
        return normalized

    def _extract_result_items(self, response: Any) -> List[Any]:
        if isinstance(response, dict):
            results = response.get("results")
            if isinstance(results, list):
                return results
            if isinstance(results, dict):
                return results.get("results", [])
            return []
        if isinstance(response, list):
            return response
        return []


_service_instance: Optional[PatientMemoryService] = None
_service_lock = Lock()


def get_patient_memory_service() -> PatientMemoryService:
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = PatientMemoryService()
    return _service_instance
