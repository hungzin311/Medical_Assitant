from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse

from .config import get_patient_memory_settings
from .memory_service import PatientMemoryServiceError, get_patient_memory_service
from .schemas import (
    PatientConditionCreate,
    PatientConversationMemoryCreate,
    PatientMemoryDeleteResponse,
    PatientMemoryListRequest,
    PatientMemoryListResponse,
    PatientMemorySearchRequest,
    PatientMemorySearchResponse,
    PatientMemoryWriteResponse,
)


router = APIRouter(prefix="/api/patient-memory", tags=["patient-memory"])


def _service():
    return get_patient_memory_service()


def _handle_service_error(exc: PatientMemoryServiceError):
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=str(exc),
    ) from exc


@router.get("/health")
def patient_memory_health():
    try:
        return _service().health()
    except PatientMemoryServiceError as exc:
        _handle_service_error(exc)


@router.post("/conditions", response_model=PatientMemoryWriteResponse)
def add_patient_condition(payload: PatientConditionCreate):
    """Store one explicit patient-condition fact.

    This endpoint stores the provided clinical fact exactly as supplied and
    does not ask the LLM to infer additional facts.
    """
    try:
        return _service().add_condition(payload)
    except PatientMemoryServiceError as exc:
        _handle_service_error(exc)


@router.post("/conversations", response_model=PatientMemoryWriteResponse)
def add_patient_conversation_memory(payload: PatientConversationMemoryCreate):
    """Store patient-condition facts extracted from conversation turns."""
    try:
        return _service().add_conversation(payload)
    except PatientMemoryServiceError as exc:
        _handle_service_error(exc)


@router.post("/search", response_model=PatientMemorySearchResponse)
def search_patient_memory(payload: PatientMemorySearchRequest):
    """Search durable condition memories for one patient."""
    try:
        return _service().search(payload)
    except PatientMemoryServiceError as exc:
        _handle_service_error(exc)


@router.get("/patients/{patient_id}/conditions", response_model=PatientMemoryListResponse)
def list_patient_conditions(
    patient_id: str,
    top_k: int = Query(50, ge=1, le=500),
    run_id: Optional[str] = None,
):
    """List stored condition memories for one patient."""
    try:
        payload = PatientMemoryListRequest(patient_id=patient_id, top_k=top_k, run_id=run_id)
        return _service().list_conditions(payload)
    except PatientMemoryServiceError as exc:
        _handle_service_error(exc)


@router.delete("/memories/{memory_id}", response_model=PatientMemoryDeleteResponse)
def delete_patient_memory(memory_id: str):
    """Delete one memory by ID."""
    try:
        return _service().delete_memory(memory_id)
    except PatientMemoryServiceError as exc:
        _handle_service_error(exc)


@router.delete("/patients/{patient_id}/conditions", response_model=PatientMemoryDeleteResponse)
def delete_all_patient_memories(
    patient_id: str,
    run_id: Optional[str] = None,
    confirm: bool = Query(False, description="Must be true to delete patient memories."),
):
    """Delete all memories for one patient, optionally scoped to run_id."""
    if not confirm:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "message": "Set confirm=true to delete patient memories.",
            },
        )
    try:
        return _service().delete_patient_memories(patient_id=patient_id, run_id=run_id)
    except PatientMemoryServiceError as exc:
        _handle_service_error(exc)


settings = get_patient_memory_settings()
app = FastAPI(title="Patient Memory Service", version="1.0.0")
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.service_host, port=settings.service_port)

