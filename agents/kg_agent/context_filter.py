from langchain_core.prompts import PromptTemplate
from typing import List, Dict
from .embedding_service import embed_text, cosine_similarity
from .cypher_query_llm import CypherQueryService

from llm_config import get_gemini_llm
import numpy as np 

class ContextFilter:
    def __init__(self):
        self.llm = get_gemini_llm(temperature=0.1)
        self.filter_prompt = PromptTemplate(
            template="""
            Bạn là chuyên gia y tế. Hãy lọc thông tin từ knowledge graph chỉ giữ lại những thông tin PHÙ HỢP với hồ sơ bệnh nhân.

            THÔNG TIN BỆNH NHÂN:
            - Tuổi: {age}
            - Giới tính: {gender}
            - Tiền sử bệnh: {medical_history}
            - Các bệnh hiện tại: {current_diseases}

            THÔNG TIN TỪ KNOWLEDGE GRAPH:
            {kg_context}

            HÃY LỌC VÀ CHỈ GIỮ LẠI:
            1. Thông tin về các bệnh phù hợp với độ tuổi và giới tính
            2. Phương pháp điều trị phù hợp với tình trạng sức khỏe hiện tại
            3. Lời khuyên phù hợp với tiền sử bệnh
            4. Loại bỏ thông tin về bệnh trẻ em nếu bệnh nhân là người lớn và ngược lại
            5. Loại bỏ thông tin không liên quan đến câu hỏi gốc

            THÔNG TIN ĐÃ LỌC:
            """,
            input_variables=["age", "gender", "medical_history", "current_diseases", "kg_context"]
        )
    
    def filter_context(self, kg_context: List[Dict], patient_info: Dict, question: str) -> str:
        """
        Lọc context từ KG dựa trên thông tin bệnh nhân
        """
        # Extract patient demographics
        age = patient_info.get('age', 'không rõ')
        gender = patient_info.get('sex', 'không rõ')
        medical_history = patient_info.get('medical_history', [])
        current_diseases = patient_info.get('diseases_active', [])
        
        # Format context for filtering
        formatted_context = self._format_kg_context(kg_context)
        
        # Apply LLM filter
        filtered_response = self.llm.invoke(
            self.filter_prompt.format(
                age=age,
                gender=gender,
                medical_history=medical_history,
                current_diseases=current_diseases,
                kg_context=formatted_context
            )
        )
        
        return filtered_response.content
    
    def _format_kg_context(self, kg_context: List[Dict]) -> str:
        """Format KG context for better readability"""
        formatted = []
        for item in kg_context:
            if isinstance(item, dict):
                formatted.append(f"- {item}")
        return "\n".join(formatted)

class ContextFilterEmbedding: 
    def __init__(self):
        self.cypher_service = CypherQueryService()

    def filter_context(self, kg_context: List[Dict], patient_context: str, question: str): 
        
        context = patient_context
        # Use embedding service
        q_vec = embed_text(context)
        
        scored = []
        for item in kg_context:
            emb = np.array(item['d']['embedding'])
            score = cosine_similarity(emb, q_vec)
            scored.append({"score": score, **item['d']})

        top5 = sorted(scored, key=lambda x: x["score"], reverse=True)[:2]

        content = [] 
        for record in top5: 
            print(record['score'])
            print(record['name'])
            # Use cypher service
            result = self.cypher_service.get_disease_info(record['name'])
            content.append(result)

        return content  

