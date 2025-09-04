from datetime import datetime
import logging
from .patient_form import PatientForm
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, VectorParams
from qdrant_client.http.models import Distance
from uuid import uuid4
from qdrant_client import models
from typing import List

class PatientFormVectorStore: 
    def __init__(self, config, collection_name: str = "patient_form"): 
        self.logger = logging.getLogger(__name__)
        self.collection_name = collection_name
        self.qdrant_url = config.rag.url
        self.qdrant_api_key = config.rag.api_key
        self.embedding_dim = 4
        self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        if not self._collection_exist():
            self._create_collection()

    
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="created_at",
            field_schema=models.PayloadSchemaType.DATETIME
        )

        # Index for patient and disease filtering
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="patient_id",
                field_schema="keyword"
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="primary_disease",
                field_schema="keyword"
            )
        except Exception:
            pass
    
    def _collection_exist(self) -> bool:
        try:
            cols = self.client.get_collections()
            return any(c.name == self.collection_name for c in cols.collections)
        except Exception as e:
            self.logger.error(f"Error checking collection exist: {e}")
            return False
    
    def _create_collection(self):
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={"dense": VectorParams(size=self.embedding_dim, distance=Distance.COSINE)}
        )

    def ingest_patient_form(self, patient_form: PatientForm):
        if not self._collection_exist():
            self._create_collection()

        # Ensure visit_id & defaults
        if not patient_form.visit_id:
            patient_form.visit_id = str(uuid4())
        if not patient_form.record_type:
            patient_form.record_type = "disease_tracking"

        payload = patient_form.model_dump()
        record_id = payload.get("visit_id", str(uuid4()))
        # Create a dummy vector for patient form (since we don't need semantic search on forms)
        dummy_vector = [0.0] * self.embedding_dim
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[{
                "id": record_id,
                "vector": {"dense": dummy_vector},
                "payload": payload
            }]
        )

    def retrieve_patient_form(self, patient_id: str):
        query_filter = Filter( 
            must = [
                FieldCondition(key = 'patient_id', match = MatchValue(value=patient_id))
            ]
        )
        patient_form, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=5,
            with_payload=True,
            order_by=models.OrderBy(
                key="created_at",
                direction=models.Direction.DESC
            )
        )
        
        return patient_form