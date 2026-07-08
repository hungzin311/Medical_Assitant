import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class PatientMemorySettings:
    collection_name: str
    history_db_path: str
    qdrant_url: Optional[str]
    qdrant_api_key: Optional[str]
    embedding_dims: int
    agent_id: str
    llm_temperature: float
    llm_max_tokens: int
    use_project_proxy: bool
    custom_instructions: str


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_patient_memory_settings() -> PatientMemorySettings:
    history_db_path = os.getenv(
        "PATIENT_MEMORY_HISTORY_DB_PATH",
        str(PROJECT_ROOT / "data" / "mem0" / "patient_memory_history.db"),
    )

    custom_instructions = """
Only extract durable patient-condition facts that are clinically useful for future
conversations. Store facts in Vietnamese when the conversation is Vietnamese.

Allowed facts:
- symptoms explicitly reported by the patient
- confirmed diagnoses or active chronic conditions
- allergies and medication intolerances
- current medications, adherence issues, side effects, and treatment response
- red flags, risk factors, recent vitals/labs, lifestyle factors relevant to care
- patient preferences that affect care communication or follow-up

Do not store:
- casual chat, greetings, or unrelated preferences
- raw identifiers, secrets, API keys, phone numbers, addresses, or raw image data
- speculative diagnoses, differential diagnoses, or possibilities unless the
  user or a clinician explicitly confirms them
- assistant disclaimers or generic medical advice

If the information is uncertain, mark it as reported/uncertain in the memory text.
Return only concise memory facts.""".strip()

    return PatientMemorySettings(
        collection_name=os.getenv("PATIENT_MEMORY_COLLECTION", "medical_patient_memories"),
        history_db_path=history_db_path,
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        embedding_dims=int(os.getenv("PATIENT_MEMORY_EMBEDDING_DIMS", "768")),
        agent_id=os.getenv("PATIENT_MEMORY_AGENT_ID", "medical_assistant_patient_memory"),
        llm_temperature=float(os.getenv("PATIENT_MEMORY_LLM_TEMPERATURE", "0.1")),
        llm_max_tokens=int(os.getenv("PATIENT_MEMORY_LLM_MAX_TOKENS", "2000")),
        use_project_proxy=_env_bool("PATIENT_MEMORY_USE_PROJECT_PROXY", False),
        custom_instructions=custom_instructions,
    )
