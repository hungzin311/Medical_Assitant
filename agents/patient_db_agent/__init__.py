import logging
from .patient_form import PatientForm
from typing import List, Dict, Any
from typing import Optional, List
from .patient_form_vectorstore import PatientFormVectorStore
from .patient_profile_store import PatientProfileStore
class PatientQueryEngine:
    """
    High-level query interface for patient database operations.
    """
    
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.patient_form_store = PatientFormVectorStore(config)
        self.profile_store = PatientProfileStore(config)
    
    ### Patient Form Store
    def ingest_patient_form(self, patient_form: PatientForm):
        return self.patient_form_store.ingest_patient_form(patient_form)

    def retrieve_patient_form(self, patient_id: str):
        return self.patient_form_store.retrieve_patient_form(patient_id)
        
    ### Patient Profile Store
    def get_patient_profile(self, patient_id: str):
        """Get patient profile with basic demographics and active diseases."""
        return self.profile_store.get_profile(patient_id)
        
    def update_profile(self, patient_data: Dict[str, Any]):
        """Create or update patient profile."""
        self.profile_store.upsert_profile(patient_data)
        
    def add_diseases_to_profile(self, patient_id: str, diseases: List[str]):
        """Add diseases to patient's active diseases list."""
        self.profile_store.add_active_diseases(patient_id, diseases)
        
    def get_patient_diseases(self, patient_id: str):
        """Get list of patient's active diseases."""
        profile = self.profile_store.get_profile(patient_id)
        return profile.get("diseases_active", []) if profile else []
