from datetime import datetime
import logging
from .patient_vectorstore import PatientVectorStore
from .patient_intake_form import PatientIntakeForm
from qdrant_client.models import Filter, FieldCondition, MatchValue
from uuid import uuid4

class PatientFormVectorStore: 
    def __init__(self, config): 
        self.patient_vector_store = PatientVectorStore(config, collection_name='patient_form')
        self.client = self.patient_vector_store.client
    
    def _collection_exist(self): 
        return self.patient_vector_store._does_collection_exist()
    
    def ingest_patient_form(self, patient_form: PatientIntakeForm):
        if not self._collection_exist():
            self.patient_vector_store._create_patient_collection()

        payload = patient_form.model_dump()
        record_id = str(uuid4())
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
                FieldCondition('patient_id', MatchValue(value=patient_id))
            ]
        )
        patient_form = self.client.query_points(
            collection_name = self.patient_vector_store.collection_name, 
            query_filter = query_filter,
            limit = 5, 
            with_payload = True, 
            order_by = 'created_at'
        )
        return patient_form.points