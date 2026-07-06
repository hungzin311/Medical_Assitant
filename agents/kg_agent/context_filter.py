from langchain_core.prompts import PromptTemplate
from typing import List, Dict
from .embedding_service import embed_text, cosine_similarity
from .cypher_query_llm import CypherQueryService
import numpy as np 

class ContextFilterEmbedding: 
    def __init__(self):
        self.cypher_service = CypherQueryService()

    def filter_context(self, kg_context: List[Dict], patient_context: str, question: str): 
        
        context = patient_context
        # Use embedding service
        q_vec = embed_text(context)
        
        scored = []
        for item in kg_context:
            if not isinstance(item, dict):
                continue
            disease = item.get('d')
            if not isinstance(disease, dict):
                continue
            embedding = disease.get('embedding')
            disease_name = disease.get('name')
            if embedding is None or not disease_name:
                continue

            num_symptoms = item.get('total_symptoms', 0)
            matched_symptoms = item.get('matched_symptoms', 0)
            matched_symptoms_list = item.get('matched_symptom_list', [])
            
            emb = np.array(embedding)
            score = cosine_similarity(emb, q_vec)
            
            # Tạo scored item với thông tin đầy đủ
            scored_item = {
                "score": score,
                "name": disease_name,
                "symptom_analysis": {
                    "total_symptoms": num_symptoms,
                    "matched_symptoms": matched_symptoms,
                    "matched_symptoms_list": matched_symptoms_list,
                }
            }
            scored.append(scored_item)

        # Sắp xếp theo score và lấy top 3
        top_results = sorted(scored, key=lambda x: x["score"], reverse=True)[:3]

        content = [] 
        for record in top_results: 
            print(f"Disease: {record['name']}, Score: {record['score']:.3f}")
            
            # Lấy thông tin chi tiết từ cypher service
            detailed_info = self.cypher_service.get_disease_info(record['name'])
            
            # Kết hợp thông tin
            result = {
                "disease_info": record,  # Thông tin bệnh + symptom analysis
                "detailed_data": detailed_info[0] if detailed_info else {}  # Thông tin chi tiết từ DB
            }
            content.append(result)

        return content  
