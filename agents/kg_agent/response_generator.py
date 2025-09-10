from context_filter import ContextFilter, ContextFilterEmbedding
from cypher_query_llm import retrieve_context_from_kg

from pathlib import Path 
import sys 
sys.path.append(str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
from config import Config
from langchain_core.prompts import PromptTemplate
from agents.patient_db_agent import PatientQueryEngine
from llm_config import *
from prompt import cypher_query
import time 
load_dotenv()

embedding_model = get_fpt_vietnamese_embedding()
graph = get_graph_db()
llm = get_gemini_llm(temperature=0.2)

config = Config()
patient_query_engine = PatientQueryEngine(config)
context_filter = ContextFilterEmbedding(embedding_model, graph, cypher_query)
patient_id = "PAT_001"

prompt = PromptTemplate(
    template="""
    Bạn là bác sĩ AI chuyên nghiệp. Hãy trả lời câu hỏi dựa trên thông tin bệnh nhân và kiến thức y tế đã được lọc.

    THÔNG TIN BỆNH NHÂN:
    {patient_info}

    KIẾN THỨC Y TẾ LIÊN QUAN (đã được lọc phù hợp với bệnh nhân):
    {filtered_context}

    CÂU HỎI: {question}

    Hãy đưa ra lời khuyên cụ thể, phù hợp với:
    - Độ tuổi và giới tính của bệnh nhân
    - Tình trạng sức khỏe hiện tại
    - Tiền sử bệnh lý
    - Chỉ sử dụng thông tin đã được lọc

    TRẢ LỜI:
    """,
    input_variables=["patient_info", "filtered_context", "question"]
)

def response_generator(question: str, use_filtering: bool = True):
    # Get patient information
    patient_record = patient_query_engine.retrieve_patient_records(patient_id, query=question)
    patient_info = patient_record[0].get('payload', {})
    
    # Get patient profile for better context
    patient_profile = patient_query_engine.get_patient_profile(patient_id)
    if patient_profile:
        patient_info.update(patient_profile)
    
    # Retrieve KG context with patient awareness
    kg_context = retrieve_context_from_kg(question)
    
    if use_filtering and isinstance(kg_context, dict) and 'result' in kg_context:
        # Apply context filtering
        filtered_context = context_filter.filter_context(
            kg_context['result'], 
            patient_info, 
            question
        )
    else:
        filtered_context = str(kg_context)
    
    # Generate response with filtered context
    start_time = time.time()
    response = llm.invoke(
        prompt.format(
            patient_info=patient_info,
            filtered_context=filtered_context,
            question=question
        )
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    return response

# Backward compatibility
def response_generator_simple(question: str):
    return response_generator(question)

def main():
    question = "Bệnh tiểu đường"
    response = response_generator(question)
    # print(response.content)
    
if __name__ == "__main__":
    main()