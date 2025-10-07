import os
import json
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.config import Config
from agents.patient_db_agent import PatientQueryEngine

def main():
    """
    Ingest patient profiles from JSON file to patient_profile collection.
    """
    # Setup
    config = Config()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize engine
    engine = PatientQueryEngine(config)
    
    # Load profiles
    profiles_path = "sample_patient_profiles.json"
    try:
        with open(profiles_path, 'r', encoding='utf-8') as f:
            profiles = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load profiles: {e}")
        return
    
    # Insert each profile
    for profile in profiles:
        try:
            engine.update_profile(profile)
            logger.info(f"Ingested profile for patient {profile.get('patient_id')}")
        except Exception as e:
            logger.error(f"Failed to ingest profile: {e}")
    
    # Verify
    for patient_id in [p.get('patient_id') for p in profiles]:
        try:
            profile = engine.get_patient_profile(patient_id)
            diseases = profile.get('diseases_active', [])
            logger.info(f"Patient {patient_id} has {len(diseases)} active diseases: {', '.join(diseases)}")
        except Exception as e:
            logger.error(f"Failed to verify profile: {e}")

if __name__ == "__main__":
    main()
