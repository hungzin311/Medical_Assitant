from pydantic import BaseModel, field_validator
from typing import Optional, List
from datetime import datetime

class PatientForm(BaseModel):
    patient_id: Optional[str] = None
    
    chief_complaint: str
    summary_text: Optional[str] = None
    
    onset_date: Optional[str] = None
    disease_duration_days: int
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
    
    visit_id: Optional[str] = None
    record_type: Optional[str] = "tracking_form"
    primary_disease: Optional[List[str]] = None

    red_flags: Optional[List[str]] = []
        
    created_at: datetime = datetime.now()
        
    # Demographics are sourced from patient_profile; not part of editable form
    
    @field_validator('severity_score')
    @classmethod
    def validate_severity_score(cls, v):
        if v < 0 or v > 10:
            raise ValueError('Severity score must be between 0 and 10')
        return v
    
    @field_validator('chief_complaint')
    @classmethod
    def validate_chief_complaint(cls, v):
        if len(v.strip()) < 4:
            raise ValueError('Chief complaint must be at least 4 characters')
        return v
    
    @field_validator('disease_duration_days')
    @classmethod
    def validate_disease_duration_days(cls, v):
        if v < 0 or v > 36500:
            raise ValueError('Disease duration days must be between 0 and 36500')
        return v

    # Primary disease must be provided (non-empty) for tracking forms
    @field_validator('primary_disease')
    @classmethod
    def validate_primary_disease(cls, v):
        if v is None or len(v) == 0:
            raise ValueError('primary_disease must contain at least 1 disease')
        # strip spaces and empty strings
        cleaned = [d.strip() for d in v if d and d.strip()]
        if len(cleaned) == 0:
            raise ValueError('primary_disease must contain at least 1 valid disease name')
        return cleaned
