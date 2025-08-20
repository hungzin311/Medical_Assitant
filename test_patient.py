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
import pprint
def demonstrate_refactored_usage():

    # Initialize core components
    config = Config()
    patient_store = PatientVectorStore(config)
    treatment_finder = TreatmentFinder(patient_store)

    # Example 2: Treatment Finding Operations
    print(f"\n{'='*50}")
    print("2. TREATMENT FINDING OPERATIONS")
    print(f"{'='*50}")
    
    try:
        patient_records = treatment_finder.find_treatment_cases(
            query="65-year-old male with chronic kidney disease stage 4 presenting with fluid overload and worsening renal function",
            limit=5, 
            patient_id = "PAT_009"
        )
        print(f"Found {len(patient_records)} similar patient records")
        pprint.pprint(patient_records)
        
    except Exception as e:
        print(f"❌ Error with treatment finder: {e}")
    
    
def main():
    """Main function to run the demonstration."""
    # set_proxy()
    try:
        demonstrate_refactored_usage()
    except Exception as e:
        logging.error(f"Error in demonstration: {e}")
        raise


if __name__ == "__main__":
    main()
