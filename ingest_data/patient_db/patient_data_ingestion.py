import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from uuid import uuid4

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import Config
from agents.patient_db_agent.patient_vectorstore import PatientVectorStore


class PatientDataIngestion:
    """
    Handles ingestion of patient medical records from various sources.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize patient vector store
        self.patient_store = PatientVectorStore(self.config)
        
    def ingest_from_json(self, json_file_path: str) -> List[str]:
        """
        Ingest patient records from JSON file.
        
        Args:
            json_file_path: Path to JSON file containing patient records
            
        Returns:
            List of ingested record IDs
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                patient_records = json.load(f)
            
            if isinstance(patient_records, dict):
                patient_records = [patient_records]
                
            # Validate and process records
            processed_records = []
            for record in patient_records:
                processed_record = self._validate_and_process_record(record)
                if processed_record:
                    processed_records.append(processed_record)
            
            # Batch ingest
            record_ids = self.patient_store.batch_ingest_patients(processed_records)
            
            self.logger.info(f"Successfully ingested {len(record_ids)} records from {json_file_path}")
            return record_ids
            
        except Exception as e:
            self.logger.error(f"Error ingesting from JSON file {json_file_path}: {e}")
            return []
    
    def _validate_and_process_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate and process a patient record to ensure it meets schema requirements.
        
        Args:
            record: Raw patient record
            
        Returns:
            Processed record or None if invalid
        """
        try:
            # Required fields
            required_fields = ['patient_id', 'summary_text']
            for field in required_fields:
                if field not in record or not record[field]:
                    self.logger.warning(f"Missing required field {field} in record")
                    return None
            
            # Set defaults
            processed = {
                'patient_id': record['patient_id'],
                'visit_id': record.get('visit_id', f"visit_{uuid4().hex[:8]}"),
                'record_type': record.get('record_type', 'encounter'),
                'timestamp': record.get('timestamp', datetime.now().isoformat()),
                'visit_date': record.get('visit_date', datetime.now().strftime('%Y-%m-%d')),
                'summary_text': record['summary_text'],
                'chief_complaint': record.get('chief_complaint', ''),
                
                # Demographics
                'age': record.get('age', 50),
                'sex': record.get('sex', 'U'),  # Unknown
                'bmi': record.get('bmi', 25.0),
                
                # Clinical data
                'vital_signs': record.get('vital_signs', {}),
                'lab_values': record.get('lab_values', {}),
                'severity_score': record.get('severity_score', 0.5),
                'severity_category': record.get('severity_category', 'moderate'),
                
                # Treatment data
                'treatments_tried': record.get('treatments_tried', []),
                'candidate_treatments': record.get('candidate_treatments', []),
                'contraindications': record.get('contraindications', []),
                
                # Comorbidities
                'comorbidities': record.get('comorbidities', []),
                'risk_factors': record.get('risk_factors', []),
                
                # Outcomes
                'primary_outcome': record.get('primary_outcome', 0.5),
                'readmission_30d': record.get('readmission_30d', False),
                'adverse_events': record.get('adverse_events', []),
                'disease_progression': record.get('disease_progression', 'stable'),
                
                # Population health
                'facility_id': record.get('facility_id', 'facility_001'),
                'provider_id': record.get('provider_id', 'provider_001'),
                'geographic_region': record.get('geographic_region', 'unknown'),
                'insurance_type': record.get('insurance_type', 'unknown'),
                
                # Risk flags
                'outbreak_risk': record.get('outbreak_risk', False),
                'unusual_presentation': record.get('unusual_presentation', False),
                'treatment_resistance': record.get('treatment_resistance', False),
                'quality_alert': record.get('quality_alert', False),
                'high_risk_patient': record.get('high_risk_patient', False),
                
                # Metadata
                'clinical_tags': record.get('clinical_tags', []),
                'data_completeness': record.get('data_completeness', 1.0),
                'research_cohort': record.get('research_cohort', []),
                'evidence_level': record.get('evidence_level', 'medium'),
                'followup_available': record.get('followup_available', False),
            }
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing record: {e}")
            return None

def main():
    # Initialize ingestion pipeline
    ingestion = PatientDataIngestion()
    
    # Ingest sample patient data
    sample_file_path = "ingest_data/patient_db/sample_patient_data.json"
    record_ids = ingestion.ingest_from_json(sample_file_path)
    print(f"Ingested {len(record_ids)} sample patient records")
    
    # Ingest risk patient data
    risk_file_path = "ingest_data/patient_db/risk_patient_data.json"
    risk_record_ids = ingestion.ingest_from_json(risk_file_path)
    print(f"Ingested {len(risk_record_ids)} risk patient records")
    
    print(f"Total records ingested: {len(record_ids) + len(risk_record_ids)}")

if __name__ == "__main__":
    main()
