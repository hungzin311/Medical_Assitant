from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConditionType(str, Enum):
    SYMPTOM = "symptom"
    DIAGNOSIS = "diagnosis"
    ALLERGY = "allergy"
    MEDICATION = "medication"
    TREATMENT_RESPONSE = "treatment_response"
    VITAL_SIGN = "vital_sign"
    RISK_FLAG = "risk_flag"
    LIFESTYLE = "lifestyle"
    GENERAL = "general"


class MemoryMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = Field(..., description="Chat role.")
    content: str = Field(..., min_length=1, description="Message content.")
    name: Optional[str] = Field(None, description="Optional actor name.")


class PatientScopedModel(BaseModel):
    patient_id: str = Field(..., min_length=1, description="Stable patient identifier.")

    @field_validator("patient_id")
    @classmethod
    def validate_patient_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("patient_id cannot be empty")
        if any(char.isspace() for char in value):
            raise ValueError("patient_id cannot contain whitespace; use '_' or '-' instead")
        return value


class PatientConditionCreate(PatientScopedModel):
    model_config = ConfigDict(use_enum_values=True)
    condition_text: str = Field(..., min_length=2, description="Clinical fact to remember.")
    condition_type: ConditionType = Field(ConditionType.GENERAL, description="Type of patient condition.")
    status: str = Field("active", description="Condition status, for example active or resolved.")
    source: str = Field("user_reported", description="Source of the remembered fact.")
    run_id: Optional[str] = Field(None, description="Optional session/run identifier.")
    primary_disease: Optional[List[str]] = Field(None, description="Related active diseases.")
    severity: Optional[float] = Field(None, ge=0, le=10, description="0-10 severity when available.")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence in the remembered fact.")
    tags: List[str] = Field(default_factory=list, description="Optional non-sensitive tags.")
    observed_at: Optional[datetime] = Field(None, description="When this condition was observed/reported.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional non-sensitive metadata.")

    @field_validator("condition_text")
    @classmethod
    def validate_condition_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("condition_text cannot be empty")
        return value


class PatientConversationMemoryCreate(PatientScopedModel):
    messages: List[MemoryMessage] = Field(..., min_length=1, description="Conversation turns to extract from.")
    run_id: Optional[str] = Field(None, description="Optional session/run identifier.")
    infer: bool = Field(True, description="Use Mem0 extraction. Set false to store raw messages.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional non-sensitive metadata.")


class PatientMemorySearchRequest(PatientScopedModel):
    query: str = Field(..., min_length=1, description="Natural-language memory query.")
    top_k: int = Field(10, ge=1, le=100, description="Maximum returned memories.")
    threshold: float = Field(0.1, ge=0, le=1, description="Minimum semantic relevance.")
    run_id: Optional[str] = Field(None, description="Restrict search to a session/run.")
    condition_types: Optional[List[ConditionType]] = Field(None, description="Restrict to condition types.")
    status: Optional[str] = Field(None, description="Restrict to a condition status.")
    metadata_filter: Dict[str, Any] = Field(default_factory=dict, description="Extra Mem0 metadata filters.")


class PatientMemoryListRequest(PatientScopedModel):
    top_k: int = Field(50, ge=1, le=500, description="Maximum returned memories.")
    run_id: Optional[str] = Field(None, description="Restrict list to a session/run.")
    condition_types: Optional[List[ConditionType]] = Field(None, description="Restrict to condition types.")
    status: Optional[str] = Field(None, description="Restrict to a condition status.")
    metadata_filter: Dict[str, Any] = Field(default_factory=dict, description="Extra Mem0 metadata filters.")


class PatientMemoryItem(BaseModel):
    id: str
    memory: str
    score: Optional[float] = None
    event: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PatientMemoryWriteResponse(BaseModel):
    status: Literal["success"]
    patient_id: str
    run_id: Optional[str] = None
    results: List[PatientMemoryItem] = Field(default_factory=list)


class PatientMemorySearchResponse(BaseModel):
    status: Literal["success"]
    patient_id: str
    query: str
    results: List[PatientMemoryItem] = Field(default_factory=list)


class PatientMemoryListResponse(BaseModel):
    status: Literal["success"]
    patient_id: str
    results: List[PatientMemoryItem] = Field(default_factory=list)


class PatientMemoryDeleteResponse(BaseModel):
    status: Literal["success"]
    message: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
