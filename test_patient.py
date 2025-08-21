from agents.patient_db_agent.patient_query_engine import PatientQueryEngine
from config import Config 
import json
import logging
import pprint
logging.basicConfig(level=logging.INFO)

# Tắt log từ httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
def main(): 
    config = Config() 
    patient_query = PatientQueryEngine(config)

    # patient_record = patient_query.retrieve_patient_records("PAT_009")
    result = patient_query.recommend_treatment("PAT_009", "65-year-old male with chronic kidney disease stage 4 presenting with fluid overload and worsening renal function")
    pprint.pprint(result)

    # with open("result.json", "w", encoding="utf-8") as f:
    #     json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":  
    main()