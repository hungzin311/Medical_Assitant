from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from agents.kg_agent import KGQueryEngine
from agents.patient_db_agent import PatientQueryEngine
from proxy_setting import set_proxy

set_proxy()

def main(): 
    from config import Config
    config = Config()
    patient_query_engine = PatientQueryEngine(config)
    kg_query_engine = KGQueryEngine(patient_query_engine)
    x = kg_query_engine.generate_medical_response("Đây là bệnh gì với các triệu chứng đau đầu, buồn nôn trong 2 ngày?", "PAT_001")
    print(x)
if __name__ == "__main__":
    main()