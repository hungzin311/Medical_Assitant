from pydantic import BaseModel, field_validator
from typing import Optional, List
from datetime import datetime

class ContactInfo(BaseModel):
    phone: str
    email: Optional[str] = None

class PatientForm(BaseModel):
    patient_id: Optional[str] = None
    full_name: Optional[str] = None
    age: int
    sex: Optional[str] = None
    bmi: Optional[float] = None
    bmi_category: Optional[str] = None
    geographic_region: Optional[str] = None
    
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
    
    red_flags: Optional[List[str]] = []
        
    contact: ContactInfo
    created_at: datetime = datetime.now()
        
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
        if len(v.strip()) < 4:
            raise ValueError('Chief complaint must be at least 4 characters')
        return v
    
    @field_validator('disease_duration_days')
    @classmethod
    def validate_disease_duration_days(cls, v):
        if v < 0 or v > 36500:
            raise ValueError('Disease duration days must be between 0 and 36500')
        return v
