from agents.patient_db_agent import PatientQueryEngine
from config import Config 
import json
import logging
import pprint
from agents.patient_db_agent import PatientQueryEngine
from agents.patient_db_agent.patient_form import PatientIntakeForm
from proxy_setting import set_proxy

set_proxy()

logging.basicConfig(level=logging.INFO)

logging.getLogger("httpx").setLevel(logging.WARNING)
def main(): 
    config = Config() 
    patient_query = PatientQueryEngine(config)
    patient_form_query = PatientQueryEngine(config) 
    logging.info(f"Patient form query: {patient_form_query.patient_form_store.patient_vector_store.collection_name}")
    logging.info(f"Patient record query: {patient_query.patient_store.collection_name}")

    patient_form = patient_form_query.retrieve_patient_form("PAT_009")
    patient_record = patient_query.retrieve_patient_records("PAT_009")

    result = patient_query.evaluate_patient_records(patient_record, patient_form)
    pprint.pprint(result)


    # with open("result.json", "w", encoding="utf-8") as f:
    #     json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":  
    main()