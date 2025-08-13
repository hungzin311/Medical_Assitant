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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import Config
from agents.patient_db_agent.patient_vectorstore import PatientVectorStore
from agents.patient_db_agent.find_treatment import TreatmentFinder
from agents.patient_db_agent.evaluate_disease import DiseaseEvaluator


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def demonstrate_refactored_usage():
    """Demonstrate usage of refactored classes."""
    print("\n" + "="*70)
    print("REFACTORED PATIENT DATABASE SYSTEM USAGE")
    print("="*70)
    
    # Initialize core components
    config = Config()
    patient_store = PatientVectorStore(config)
    treatment_finder = TreatmentFinder(patient_store)
    disease_evaluator = DiseaseEvaluator(patient_store)
    
    print(f"\n✅ Initialized core components:")
    print(f"   - PatientVectorStore: {patient_store.__class__.__name__}")
    print(f"   - TreatmentFinder: {treatment_finder.__class__.__name__}")
    print(f"   - DiseaseEvaluator: {disease_evaluator.__class__.__name__}")
    

    # Example 2: Treatment Finding Operations
    print(f"\n{'='*50}")
    print("2. TREATMENT FINDING OPERATIONS")
    print(f"{'='*50}")
    
    try:
        # Search for patient records
        print("Searching for similar patient records...")
        records = treatment_finder.retrieve_patient_records(
            query="male presenting with unusual respiratory illness. High fever",
            demographic_filters={"age_range": (30, 50), "sex": "F"},
            clinical_filters={"comorbidities": ["hypertension"]},
            limit=5
        )
        print(f"Found {len(records)} similar patient records")
        
        # Find treatment cases
        print("Finding similar treatment cases...")
        treatment_cases = treatment_finder.find_treatment_cases(
            query="chest pain cardiovascular symptoms",
            candidate_treatments=["aspirin", "beta-blocker", "ACE inhibitor"],
            age_range=(40, 50),
            comorbidities=["hypertension"],
            limit=5
        )
        print(f"Found {len(treatment_cases)} similar treatment cases")
        
    except Exception as e:
        print(f"❌ Error with treatment finder: {e}")
    
    # # Example 3: Disease Evaluation Operations
    # print(f"\n{'='*50}")
    # print("3. DISEASE EVALUATION OPERATIONS")
    # print(f"{'='*50}")
    
    # try:
    #     # Monitor population patterns
    #     print("Monitoring population patterns...")
    #     flagged_cases = disease_evaluator.monitor_population_patterns(
    #         time_window="90d",  # Use longer time window to include sample data
    #         geographic_region="Northeast",
    #         risk_flags=["outbreak", "unusual"],
    #         limit=10
    #     )
    #     print(f"Found {len(flagged_cases)} flagged cases for monitoring")
        
    #     # Evaluate outbreak risk
    #     print("Evaluating outbreak risk...")
    #     outbreak_assessment = disease_evaluator.evaluate_outbreak_risk(
    #         geographic_region="Northeast",
    #         time_window="90d"  # Use longer time window to include sample data
    #     )
    #     print(f"Outbreak risk level: {outbreak_assessment.get('risk_level', 'unknown')}")
    #     print(f"Total cases analyzed: {outbreak_assessment.get('total_cases', 0)}")
        
    #     # Get population health summary
    #     print("Getting population health summary...")
    #     health_summary = disease_evaluator.get_population_health_summary(
    #         geographic_region="Northeast",
    #         time_window="90d"  # Use longer time window to include sample data
    #     )
    #     print(f"Population summary - Total patients: {health_summary.get('total_patients', 0)}")
    #     print(f"Average risk score: {health_summary.get('average_risk_score', 0)}")
        
    except Exception as e:
        print(f"❌ Error with disease evaluator: {e}")


def main():
    """Main function to run the demonstration."""
    setup_logging()
    
    try:
        demonstrate_refactored_usage()
    except Exception as e:
        logging.error(f"Error in demonstration: {e}")
        raise


if __name__ == "__main__":
    main()
