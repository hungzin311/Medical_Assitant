import os 
from dotenv import load_dotenv
from agents.agent_decision import config
from llm_config import * 
from .cypher_query_llm import retrieve_context_from_kg
from .context_filter import ContextFilter
from langchain_core.prompts import PromptTemplate
from agents.patient_db_agent import PatientQueryEngine
load_dotenv()

llm = get_gemini_llm(temperature=0.2)
patient_query_engine = PatientQueryEngine(config)
context_filter = ContextFilter()
patient_id = "PAT_001"

# Enhanced prompt with better context integration
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
    kg_context = retrieve_context_from_kg(question, patient_info if use_filtering else None)
    
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
    response = llm.invoke(
        prompt.format(
            patient_info=patient_info,
            filtered_context=filtered_context,
            question=question
        )
    )
    return response

# Backward compatibility
def response_generator_simple(question: str):
    """Simple version without filtering for comparison"""
    return response_generator(question, use_filtering=False)
