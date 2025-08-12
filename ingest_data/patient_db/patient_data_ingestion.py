"""
Patient Data Ingestion Pipeline

This module handles the ingestion of patient medical records into the Qdrant vector database
for the three core problems:
1. Patient record retrieval
2. Treatment optimization 
3. Early warning systems
"""

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
    
    def ingest_from_csv(self, csv_file_path: str, column_mapping: Optional[Dict[str, str]] = None) -> List[str]:
        """
        Ingest patient records from CSV file.
        
        Args:
            csv_file_path: Path to CSV file
            column_mapping: Optional mapping of CSV columns to patient record fields
            
        Returns:
            List of ingested record IDs
        """
        try:
            df = pd.read_csv(csv_file_path)
            
            # Apply column mapping if provided
            if column_mapping:
                df = df.rename(columns=column_mapping)
            
            # Convert DataFrame to patient records
            patient_records = []
            for _, row in df.iterrows():
                record = self._dataframe_row_to_patient_record(row)
                if record:
                    patient_records.append(record)
            
            # Validate and process records
            processed_records = []
            for record in patient_records:
                processed_record = self._validate_and_process_record(record)
                if processed_record:
                    processed_records.append(processed_record)
            
            # Batch ingest
            record_ids = self.patient_store.batch_ingest_patients(processed_records)
            
            self.logger.info(f"Successfully ingested {len(record_ids)} records from {csv_file_path}")
            return record_ids
            
        except Exception as e:
            self.logger.error(f"Error ingesting from CSV file {csv_file_path}: {e}")
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
    
    def _dataframe_row_to_patient_record(self, row: pd.Series) -> Dict[str, Any]:
        """
        Convert DataFrame row to patient record format.
        
        Args:
            row: Pandas Series representing a row
            
        Returns:
            Patient record dictionary
        """
        record = {}
        
        # Map common column names
        column_mapping = {
            'id': 'patient_id',
            'patient_id': 'patient_id',
            'text': 'summary_text',
            'summary': 'summary_text',
            'summary_text': 'summary_text',
            'complaint': 'chief_complaint',
            'chief_complaint': 'chief_complaint',
            'age': 'age',
            'gender': 'sex',
            'sex': 'sex',
            'bmi': 'bmi',
            'comorbidities': 'comorbidities',
            'treatments': 'treatments_tried',
            'outcome': 'primary_outcome',
        }
        
        for col_name, value in row.items():
            if pd.isna(value):
                continue
                
            # Map column name
            field_name = column_mapping.get(col_name.lower(), col_name.lower())
            
            # Process value based on field type
            if field_name in ['comorbidities', 'treatments_tried', 'clinical_tags']:
                # Handle list fields
                if isinstance(value, str):
                    record[field_name] = [item.strip() for item in value.split(',') if item.strip()]
                else:
                    record[field_name] = value
            elif field_name in ['age', 'bmi', 'severity_score', 'primary_outcome']:
                # Handle numeric fields
                try:
                    record[field_name] = float(value)
                except:
                    record[field_name] = 0.0
            elif field_name in ['outbreak_risk', 'unusual_presentation', 'treatment_resistance', 
                              'quality_alert', 'high_risk_patient', 'readmission_30d']:
                # Handle boolean fields
                if isinstance(value, str):
                    record[field_name] = value.lower() in ['true', '1', 'yes']
                else:
                    record[field_name] = bool(value)
            else:
                # Handle string fields
                record[field_name] = str(value)
        
        return record
    
    

def main():
    """Example usage of the patient data ingestion pipeline."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize ingestion pipeline
    ingestion = PatientDataIngestion()
    
    # Generate and ingest synthetic data
    print("Generating synthetic patient data...")
    record_ids = ingestion.ingest_synthetic_data(num_records=50)
    print(f"Ingested {len(record_ids)} synthetic patient records")
    
    # Get collection stats
    stats = ingestion.patient_store.get_collection_stats()
    print(f"Collection statistics: {stats}")


if __name__ == "__main__":
    main()
