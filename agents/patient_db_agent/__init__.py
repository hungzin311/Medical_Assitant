import logging
from agents.patient_db_agent.disease_evaluation import DiseaseEvaluation
from .find_treatment import TreatmentFinder
from typing import List, Dict, Any
from agents.patient_db_agent.patient_vectorstore import PatientVectorStore
from pydantic import BaseModel, field_validator
from typing import Optional

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

class ContactInfo(BaseModel):
    phone: str
    email: Optional[str] = None

class RedFlags(BaseModel):
    severe_chest_pain: bool = False
    severe_breathing_difficulty: bool = False
    neurological_deficit: bool = False
    confusion_or_fainting: bool = False
    uncontrolled_bleeding: bool = False
    fever_high_persistent: bool = False
    persistent_vomiting_dehydration: bool = False
    severe_abdominal_pain: bool = False

class PatientIntakeForm(BaseModel):
    patient_id: Optional[str] = None
    full_name: Optional[str] = None
    age: int
    sex: str
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    province: Optional[str] = None
    
    chief_complaint: str
    summary_text: Optional[str] = None
    
    onset_date: Optional[str] = None
    duration_days: int
    severity_score: float
    
    course: Optional[str] = None
    aggravating_relieving: Optional[str] = None
    associated_symptoms: Optional[List[str]] = []
    location: Optional[str] = None
    
    comorbidities: Optional[List[str]] = []
    allergies: Optional[str] = None
    
    current_medications: Optional[str] = None
    treatments_tried: Optional[str] = None
    contraindications: Optional[str] = None
    
    smoking_status: Optional[str] = None
    alcohol_use: Optional[str] = None
    occupational_exposure: Optional[str] = None
    pregnancy_status: Optional[str] = None
    
    red_flags: Optional[RedFlags] = RedFlags()
    
    expectation: Optional[str] = None
    
    contact: ContactInfo
    
    consent: bool = True
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 0 or v > 120:
            raise ValueError('Age must be between 0 and 120')
        return v
    
    @field_validator('sex')
    @classmethod
    def validate_sex(cls, v):
        if v not in ['male', 'female', 'other']:
            raise ValueError('Sex must be male, female, or other')
        return v
    
    @field_validator('severity_score')
    @classmethod
    def validate_severity_score(cls, v):
        if v < 0 or v > 10:
            raise ValueError('Severity score must be between 0 and 10')
        return v
    
    @field_validator('chief_complaint')
    @classmethod
    def validate_chief_complaint(cls, v):
        if len(v.strip()) < 5:
            raise ValueError('Chief complaint must be at least 5 characters')
        return v
    
    @field_validator('duration_days')
    @classmethod
    def validate_duration_days(cls, v):
        if v < 0 or v > 36500:
            raise ValueError('Duration days must be between 0 and 36500')
        return v
