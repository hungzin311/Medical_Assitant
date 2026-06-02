import logging
import json
import re
from typing import List, Dict, Any, Optional
from utils.prompt import rag_agent_mcq_evaluation_prompt
from utils.streaming import invoke_with_streaming

class ResponseGenerator:
    def __init__(self, config):
        
        self.logger = logging.getLogger(__name__)
        self.response_generator_model = config.rag.llm
        self.include_sources = getattr(config.rag, "include_sources", True)

    def _build_prompt(
            self,
            query: str, 
            context: str,
            patient_context: str = "",
            chat_history: Optional[List[Dict[str, str]]] = None
        ) -> str:

        prompt = f"""Bạn là bác sĩ chuyên khoa. Hãy áp dụng tư duy lâm sàng từng bước để phân tích tình trạng bệnh nhân dựa trên TÀI LIỆU Y TẾ được truy xuất.

THÔNG TIN BỆNH NHÂN:
{patient_context}

TÀI LIỆU Y TẾ ĐƯỢC TRUY XUẤT:
{context}

CÂU HỎI/MIÊU TẢ TRIỆU CHỨNG CỦA BỆNH NHÂN: {query}

LỊCH SỬ HỘI THOẠI: {chat_history}

HƯỚNG DẪN QUAN TRỌNG:
- Các tài liệu y tế được cung cấp chứa thông tin từ các nguồn uy tín
- Chỉ sử dụng thông tin có trong các tài liệu được truy xuất
- Không tự tạo ra thông tin không có trong tài liệu

PHÂN TÍCH THEO CHAIN OF THOUGHT:

## BƯỚC 1: PHÂN TÍCH TÀI LIỆU & ĐỐI CHIẾU
1A) Phân tích các tài liệu liên quan:
- Xác định thông tin y tế có liên quan đến câu hỏi/triệu chứng
- Liệt kê các bệnh/tình trạng được đề cập trong tài liệu
- Đối chiếu triệu chứng của bệnh nhân với thông tin trong tài liệu
- Xác định mức độ tương đồng giữa tình trạng bệnh nhân và thông tin tài liệu

1B) Đánh giá độ tin cậy:
- Mức độ phù hợp của thông tin với trường hợp cụ thể
- Tính đầy đủ của thông tin để đưa ra kết luận
- Các thông tin quan trọng còn thiếu

## BƯỚC 2: QUYẾT ĐỊNH Y KHOA
- ENOUGH_INFO nếu tài liệu cung cấp đủ thông tin để trả lời câu hỏi một cách có căn cứ
- NOT_ENOUGH_INFO nếu thông tin chưa đủ hoặc không có trong tài liệu

## BƯỚC 3: HÀNH ĐỘNG PHÙ HỢP
- Nếu ENOUGH_INFO: đưa câu trả lời dựa trên tài liệu, tư vấn phù hợp
- Nếu NOT_ENOUGH_INFO: nêu rõ thông tin nào còn thiếu, khuyến nghị tham khảo thêm hoặc khám bác sĩ

NGUYÊN TẮC AN TOÀN:
- Chỉ dựa vào thông tin có trong tài liệu được truy xuất
- Không dùng ngôn ngữ khẳng định tuyệt đối; dùng "có thể", "theo tài liệu", "dựa trên thông tin"
- Khuyến cáo khám bác sĩ khi cần thiết
- Đảm bảo thông tin chính xác và có nguồn gốc

TRẢ LỜI THEO FORMAT JSON:
{{
  "step3_action": {{
    "content": "Nội dung chính trả lời bệnh nhân với format đẹp, có cấu trúc rõ ràng",
    "information_gaps": "thông_tin_còn_thiếu",  // chỉ khi NOT_ENOUGH_INFO
    "confidence": "0.0 - 1.0" // độ tự tin của mô hình cho câu trả lời chính
  }}
}}

HÃY PHÂN TÍCH:"""

        return prompt

    def generate_response(
            self,
            query: str,
            retrieved_docs: List[Dict[str, Any]],
            patient_context: str = "",
            chat_history: Optional[List[Dict[str, str]]] = None,
        ) -> Dict[str, Any]:
    
        try:
           
            # Extract content from documents for context
            doc_texts = [doc["content"] for doc in retrieved_docs]            
            context = "\n\n===DOCUMENT SECTION===\n\n".join(doc_texts)
            
            # Build the prompt
            prompt = self._build_prompt(query, context, patient_context, chat_history)            
            response = invoke_with_streaming(self.response_generator_model, prompt)
            
            # Parse JSON response and extract content (similar to KG agent)
            try:
                if hasattr(response, 'content'):
                    response_text = response.content.strip()
                else:
                    response_text = str(response).strip()
                
                match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
                if match:
                    response_text = match.group(1)  # chỉ lấy phần { ... }
                
                response_json = json.loads(response_text)
                
                # Extract content from step3_action (similar to KG agent)
                content = response_json.get("step3_action", {}).get("content", "Không có thông tin liên quan từ tài liệu.")
                confidence = response_json.get('step3_action', {}).get('confidence', 0.0)
                
                # Ensure we have good structured content
                if not content or len(content.strip()) < 10:
                    content = "Không có đủ thông tin trong tài liệu để trả lời câu hỏi này một cách đầy đủ."
                
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                self.logger.warning(f"Error parsing JSON response: {e}")
                # Fallback to original content if JSON parsing fails
                if hasattr(response, 'content'):
                    content = response.content
                else:
                    content = str(response)
            
            # Extract sources for citation
            sources = self._extract_sources(retrieved_docs) if hasattr(self, 'include_sources') and self.include_sources else []
            
            # Add safety disclaimer
            safety_disclaimer = "\n\n⚠️ **Lưu ý quan trọng:** Thông tin trên chỉ mang tính chất tham khảo và được tạo ra bởi AI. Đây không phải là chẩn đoán y tế chính thức. Bạn nên đi khám bác sĩ chuyên khoa sớm nhất có thể để được thăm khám và điều trị phù hợp."
            
            if hasattr(self, 'include_sources') and self.include_sources and sources:
                response_with_source = content + safety_disclaimer + "\n\n##### Tài liệu nguồn:"
                for current_source in sources:
                    source_path = current_source['path']
                    source_title = current_source['title']
                    response_with_source += f"\n- [{source_title}]({source_path})"
            else:
                response_with_source = content + safety_disclaimer
            
            # Format final response - ensure we always return a string in the response key
            result = {
                "response": response_with_source,
                "sources": sources,
                "confidence": confidence
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                "response": "Tôi xin lỗi, nhưng tôi đã gặp lỗi khi tạo câu trả lời. Vui lòng thử diễn đạt lại câu hỏi của bạn.",
                "sources": [],
                "confidence": 0.0
            }

    def _extract_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        sources = []
        seen_sources = set()  # Track unique sources to avoid duplicates
        
        for doc in documents:
            # Extract source and source_path
            source = doc.get("source")
            source_path = doc.get("source_path")
            
            if not source:
                continue
                
            # Create a unique identifier for this source
            source_id = f"{source}|{source_path}"
            
            if source_id in seen_sources:
                continue
                
            source_info = {
                "title": source,
                "path": source_path,
                "score": doc.get("combined_score", doc.get("rerank_score", doc.get("score", 0.0)))
            }
            
            sources.append(source_info)
            seen_sources.add(source_id)
        
        # Sort sources by score from highest to lowest
        sources.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Format the final sources list, removing the scores which were just used for sorting
        formatted_sources = []
        for source in sources:
            formatted_source = {
                "title": source["title"],
                "path": source["path"]
            }
            formatted_sources.append(formatted_source)
            
        return formatted_sources

    def generate_response_benchmark(
            self,
            question: str, 
            choices: List[str],
            retrieved_docs: List[Dict[str, Any]],
        ) -> Dict[str, Any]:
    
        try:
            doc_texts = [f"[{doc['disease_name']}]({doc['description']}) \n {doc['cause']} \n {doc['symptom']}" for doc in retrieved_docs]
            prompt = rag_agent_mcq_evaluation_prompt.format(question=question, choices=choices, context = doc_texts)
            response = self.response_generator_model.invoke(prompt)

            try:
                response_text = response.content.strip()
                
                match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
                if match:
                    response_text = match.group(1)  
                
                response_json = json.loads(response_text)
                
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                self.logger.warning(f"Error parsing JSON response: {e}")
                return {
                    "answer_index": None,
                    "not_enough_info": True,
                    "confidence": 0.0
                }
            
            return response_json
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            
            return {
                "answer_index": None,
                "not_enough_info": True,
                "confidence": 0.0
            }
