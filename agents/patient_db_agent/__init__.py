import logging
from .disease_evaluation import DiseaseEvaluation
from .patient_form import PatientForm
from .find_treatment import TreatmentFinder
from typing import List, Dict, Any
from .patient_vectorstore import PatientVectorStore
from typing import Optional, List
from .ingest_patient_form import PatientFormVectorStore
class PatientQueryEngine:
    """
    High-level query interface for patient database operations.
    """
    
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.patient_store = PatientVectorStore(config, collection_name='medical_records')
        self.treatment_finder = TreatmentFinder(self.patient_store)
        self.disease_evaluator = DiseaseEvaluation(self.patient_store)
        self.config = config
        self.patient_form_store = PatientFormVectorStore(config)
    
    ### Patient Store

    def retrieve_patient_records(self, patient_id: str):
        return self.disease_evaluator.retrieve_patient_records(patient_id)

    def evaluate_patient_records(self, patient_record: List[Dict[str, Any]], patient_form: List[Dict[str, Any]]):
        return self.disease_evaluator.evaluate_based_record_llm(patient_record, patient_form)

    def recommend_treatment(self, patient_id: str, query: str):
        
        patient_record = self.retrieve_patient_records(patient_id)
        
        # get attributes from patient record
        comorbidities = patient_record[0].get('comorbidities', [])
        candidate_treatments = patient_record[0].get('candidate_treatments', [])
        age_range_str = patient_record[0].get('age_group', '0-70').split('-')
        age_range = (int(age_range_str[0]), int(age_range_str[1]))

        # find treatment cases
        reference_records = self.treatment_finder.find_treatment_cases(query, patient_id, candidate_treatments, age_range, comorbidities)
        return self.treatment_finder.recommend_treatment_llm(patient_record, reference_records)

    ### Patient Form Store
    def ingest_patient_form(self, patient_form: PatientForm):
        return self.patient_form_store.ingest_patient_form(patient_form)

    def retrieve_patient_form(self, patient_id: str):
        return self.patient_form_store.retrieve_patient_form(patient_id)
