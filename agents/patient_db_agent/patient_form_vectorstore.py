from datetime import datetime
import logging
from .patient_vectorstore import PatientVectorStore
from .patient_form import PatientForm
from qdrant_client.models import Filter, FieldCondition, MatchValue
from uuid import uuid4
from qdrant_client import models

class PatientFormVectorStore: 
    def __init__(self, config): 
        self.patient_vector_store = PatientVectorStore(config, collection_name='patient_form')
        self.client = self.patient_vector_store.client
        self.client.create_payload_index(
            collection_name=self.patient_vector_store.collection_name,
            field_name="created_at",
            field_schema=models.PayloadSchemaType.DATETIME
        )

        # Index for patient and disease filtering
        try:
            self.client.create_payload_index(
                collection_name=self.patient_vector_store.collection_name,
                field_name="patient_id",
                field_schema="keyword"
            )
            self.client.create_payload_index(
                collection_name=self.patient_vector_store.collection_name,
                field_name="primary_disease",
                field_schema="keyword"
            )
        except Exception:
            pass
    
    def _collection_exist(self): 
        return self.patient_vector_store._does_collection_exist()
    
    def ingest_patient_form(self, patient_form: PatientForm):
        if not self._collection_exist():
            self.patient_vector_store._create_patient_collection()

        # Ensure visit_id & defaults
        if not patient_form.visit_id:
            patient_form.visit_id = str(uuid4())
        if not patient_form.record_type:
            patient_form.record_type = "tracking_form"

        payload = patient_form.model_dump()
        record_id = payload.get("visit_id", str(uuid4()))
        # Create a dummy vector for patient form (since we don't need semantic search on forms)
        dummy_vector = [0.0] * self.patient_vector_store.embedding_dim
        
        self.client.upsert(
            collection_name=self.patient_vector_store.collection_name,
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
            collection_name=self.patient_vector_store.collection_name,
            scroll_filter=query_filter,
            limit=5,
            with_payload=True,
            order_by=models.OrderBy(
                key="created_at",
                direction=models.Direction.DESC  # Mới nhất trước
            )
        )
        
        print(type(patient_form))
        print(len(patient_form))

        return patient_form