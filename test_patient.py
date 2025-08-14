"""
Example usage of the refactored Patient Database System

This script demonstrates how to use the refactored classes directly:
- PatientVectorStore: Core vector operations
- TreatmentFinder: Treatment-related operations
- DiseaseEvaluator: Population health monitoring
"""

import sys
import logging
from pathlib import Path
from config import Config  
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from proxy_setting import set_proxy
from config import Config
import json
from agents.patient_db_agent.patient_vectorstore import PatientVectorStore
from agents.patient_db_agent.find_treatment import TreatmentFinder

def demonstrate_refactored_usage():
    """Demonstrate usage of refactored classes."""
    print("\n" + "="*70)
    print("REFACTORED PATIENT DATABASE SYSTEM USAGE")
    print("="*70)
    
    # Initialize core components
    config = Config()
    patient_store = PatientVectorStore(config)
    treatment_finder = TreatmentFinder(patient_store)

    llm = config.rag.response_generator_model
    # Example 2: Treatment Finding Operations
    print(f"\n{'='*50}")
    print("2. TREATMENT FINDING OPERATIONS")
    print(f"{'='*50}")
    
    try:
        # Search for patient records
        print("Searching for similar patient records...")
        patient_records, reference_records = treatment_finder.retrieve_patient_records(
            query="male presenting with unusual respiratory illness. High fever",
            demographic_filters={"age_range": (30, 50), "sex": "F"},
            clinical_filters={"comorbidities": ["hypertension"]},
            limit=5
        )
        response = llm.invoke(f"Here is the patient record: {json.dumps(patient_records)}")
        print(response.content)

        print(f"Found {len(patient_records)} similar patient records")
        
    except Exception as e:
        print(f"❌ Error with treatment finder: {e}")
    
    
def main():
    """Main function to run the demonstration."""
    set_proxy()
    try:
        demonstrate_refactored_usage()
    except Exception as e:
        logging.error(f"Error in demonstration: {e}")
        raise


if __name__ == "__main__":
    main()
