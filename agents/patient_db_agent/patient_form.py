from pydantic import BaseModel, field_validator
from typing import Optional, List, Dict, Union
from datetime import datetime
from enum import Enum

class VisitType(str, Enum):
    ROUTINE_FOLLOWUP = "routine_followup"
    URGENT_FOLLOWUP = "urgent_followup" 
    EMERGENCY = "emergency"
    MEDICATION_REVIEW = "medication_review"
    LAB_REVIEW = "lab_review"

class TreatmentResponse(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good" 
    FAIR = "fair"
    POOR = "poor"
    NO_RESPONSE = "no_response"

class PatientForm(BaseModel):
    # Core identification
    patient_id: Optional[str] = None
    visit_id: Optional[str] = None
    visit_type: VisitType = VisitType.ROUTINE_FOLLOWUP
    
    # Disease being tracked
    primary_disease: List[str]  # Required for tracking
    disease_status: Optional[str] = None  # "stable", "improving", "worsening", "resolved"
    
    # Visit context
    days_since_last_visit: Optional[int] = None
    
    # Current status assessment
    current_symptoms: Optional[List[str]] = []
    symptom_severity_change: Optional[str] = None  # "better", "same", "worse"
    overall_severity_score: float = 0.0  # 0-10 scale
    
    # Vital signs & measurements
    vital_signs: Optional[Dict[str, Union[float, str]]] = {}  # BP, HR, temp, weight, etc.
    lab_values: Optional[Dict[str, Union[float, str]]] = {}   # glucose, HbA1c, etc.
    
    # Treatment tracking
    current_medications: Optional[List[Dict[str, str]]] = []  # name, dose, frequency, start_date
    medication_changes: Optional[List[str]] = []  # "started X", "stopped Y", "increased Z"
    treatment_response: Optional[TreatmentResponse] = None
    side_effects: Optional[List[str]] = []
    
    # Functional status
    quality_of_life_score: Optional[float] = None  # 0-10 scale
    
    # Compliance & lifestyle
    medication_compliance: Optional[str] = None  # "excellent", "good", "fair", "poor"
    lifestyle_changes: Optional[List[str]] = []
    diet_compliance: Optional[str] = None
    exercise_compliance: Optional[str] = None
    
    # Clinical notes
    provider_notes: Optional[str] = None
    patient_concerns: Optional[str] = None
    
    # Goals & plans
    treatment_goals_met: Optional[List[str]] = []
    new_treatment_goals: Optional[List[str]] = []
    next_visit_plan: Optional[str] = None
    
    # Risk assessment
    red_flags: Optional[List[str]] = []
    risk_level: Optional[str] = None  # "low", "moderate", "high"
    
    # Comorbidities tracking
    comorbidities: Optional[List[str]] = []
    new_comorbidities: Optional[List[str]] = []
    
    # System fields
    record_type: str = "disease_tracking"
    created_at: datetime = datetime.now()
    provider_id: Optional[str] = None
    
    # Validators
    @field_validator('overall_severity_score')
    @classmethod
    def validate_severity_score(cls, v):
        if v < 0 or v > 10:
            raise ValueError('Severity score must be between 0 and 10')
        return v
    
    @field_validator('primary_disease')
    @classmethod
    def validate_primary_disease(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one primary disease must be specified for tracking')
        return [d.strip() for d in v if d and d.strip()]
    
    @field_validator('quality_of_life_score')
    @classmethod
    def validate_qol_score(cls, v):
        if v is not None and (v < 0 or v > 10):
            raise ValueError('Quality of life score must be between 0 and 10')
        return v