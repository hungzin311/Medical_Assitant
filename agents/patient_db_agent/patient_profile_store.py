from typing import Dict, Any, List, Optional
import uuid
import logging
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import Filter, FieldCondition, MatchValue, PayloadSchemaType
from ..qdrant_client_manager import QdrantClientManager


class PatientProfileStore:
    """Lightweight store holding one document per patient with demographics and disease lists."""

    def __init__(self, config, collection_name: str = "patient_profile"):
        self.logger = logging.getLogger(__name__)
        self.collection_name = collection_name
        self.qdrant_url = config.rag.url
        self.qdrant_api_key = config.rag.api_key
        self.embedding_dim = 4
        
        # Initialize singleton client manager
        self.client_manager = QdrantClientManager(config)
        self.client = self.client_manager.client
        if not self._collection_exist():
            self._create_collection()

    def _collection_exist(self) -> bool:
        return self.client_manager.does_collection_exist(self.collection_name)

    def _create_collection(self):
        try:
            vectors_config = {"dense": VectorParams(size=self.embedding_dim, distance=Distance.COSINE)}
            self.client_manager.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config
            )
            # basic indexes
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="patient_id",
                field_schema="keyword"
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="diseases_active",
                field_schema="keyword"
            )
        except Exception as e:
            self.logger.error(f"Error creating profile collection: {e}")

    # ---------------- public methods -----------------
    def upsert_profile(self, payload: Dict[str, Any]):
        """Insert or update patient profile. patient_id required."""
        patient_id = payload.get("patient_id")
        pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, patient_id)) 
        if not pid:
            raise ValueError("patient_id missing in profile payload")
        vector = [0.0] * self.embedding_dim
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[{"id": pid, "vector": {"dense": vector}, "payload": payload}]
            )
        except Exception as e:
            self.logger.error(f"Error upserting profile: {e}")

    def patch_profile(self, patient_id: str, patch: Dict[str, Any]):
        """Partial update."""
        try:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=patch,
                points_selector=[patient_id]
            )
        except Exception as e:
            self.logger.error(f"Error patching profile: {e}")

    def get_profile(self, patient_id: str) -> Optional[Dict[str, Any]]:
        try:
            pts, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=[FieldCondition(key="patient_id", match=MatchValue(value=patient_id))]),
                limit=1,
                with_payload=True
            )
            if pts:
                return pts[0].payload
        except Exception as e:
            self.logger.error(f"Error get profile: {e}")
        return None

    def add_active_diseases(self, patient_id: str, diseases: List[str]):
        prof = self.get_profile(patient_id) or {"patient_id": patient_id, "diseases_active": []}
        current = set(prof.get("diseases_active", []))
        current.update(diseases)
        prof["diseases_active"] = list(current)
        self.upsert_profile(prof)
