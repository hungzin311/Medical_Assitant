import logging
from agents.patient_db_agent.disease_evaluation import DiseaseEvaluation
from .find_treatment import TreatmentFinder
from typing import List, Dict, Any, Tuple
from agents.patient_db_agent.patient_vectorstore import PatientVectorStore

class PatientQueryEngine:
    """
    High-level query interface for patient database operations.
    """
    
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.patient_store = PatientVectorStore(config)
        self.treatment_finder = TreatmentFinder(self.patient_store)
        self.disease_evaluator = DiseaseEvaluation(self.patient_store)
        self.config = config

    def retrieve_patient_records(self, patient_id: str):
        return self.disease_evaluator.retrieve_patient_records(patient_id)

    def evaluate_patient_records(self, patient_record: List[Dict[str, Any]]):
        return self.disease_evaluator.evaluate_based_record_llm(patient_record)

    def recommend_treatment(self, patient_id: str, query: str):
        """
        Recommend treatment for a patient based on their medical history and query.
        """
        patient_record = self.retrieve_patient_records(patient_id)
        
        # get attributes from patient record
        comorbidities = patient_record[0].get('comorbidities', [])
        candidate_treatments = patient_record[0].get('candidate_treatments', [])
        age_range_str = patient_record[0].get('age_group', '0-70').split('-')
        age_range = (int(age_range_str[0]), int(age_range_str[1]))

        # find treatment cases
        reference_records = self.treatment_finder.find_treatment_cases(query, patient_id, candidate_treatments, age_range, comorbidities)
        return self.treatment_finder.recommend_treatment_llm(patient_record, reference_records)

    