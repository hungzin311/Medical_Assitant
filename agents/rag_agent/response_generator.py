import logging
from typing import List, Dict, Any, Optional, Union

class ResponseGenerator:
    """
    Generates responses based on retrieved context and user query.
    """
    def __init__(self, config):
        
        self.logger = logging.getLogger(__name__)
        self.response_generator_model = config.rag.response_generator_model
        self.include_sources = getattr(config.rag, "include_sources", True)

    def _build_prompt(
            self,
            query: str, 
            context: str,
            chat_history: Optional[List[Dict[str, str]]] = None
        ) -> str:

        response_format_instructions = """Hướng dẫn:
        1. Trả lời câu hỏi CHỈ dựa trên thông tin được cung cấp trong ngữ cảnh.
        2. Nếu ngữ cảnh không chứa thông tin liên quan để trả lời câu hỏi, hãy nêu rõ: "Tôi không có đủ thông tin để trả lời câu hỏi này dựa trên ngữ cảnh được cung cấp."
        3. Không sử dụng kiến thức trước đó không có trong ngữ cảnh.
        5. Hãy ngắn gọn và chính xác.
        6. Cung cấp câu trả lời có cấu trúc tốt với tiêu đề, tiêu đề phụ dựa trên kiến thức đã truy xuất. Giữ các tiêu đề và tiêu đề phụ có kích thước nhỏ.
        7. Chỉ cung cấp các phần có ý nghĩa trong câu trả lời của chatbot. Ví dụ, không đề cập rõ ràng đến tài liệu tham khảo.
        8. Nếu liên quan đến các giá trị, hãy đảm bảo trả lời với các giá trị chính xác có trong ngữ cảnh. Không tự tạo ra giá trị.
        9. Không lặp lại câu hỏi trong câu trả lời."""
            
        # Build the prompt
        prompt = f"""Bạn là một trợ lý y tế cung cấp thông tin chính xác dựa trên các nguồn y tế đã được xác minh.

        Đây là những tin nhắn gần đây từ cuộc trò chuyện của chúng ta:
        
        {chat_history}

        Người dùng đã đặt câu hỏi sau:
        {query}

        Tôi đã truy xuất những thông tin sau để giúp trả lời câu hỏi này:

        {context}

        {response_format_instructions}

        Dựa trên thông tin được cung cấp, hãy trả lời câu hỏi của người dùng một cách kỹ lưỡng nhưng ngắn gọn.
        Nếu thông tin không chứa câu trả lời, hãy thừa nhận giới hạn của thông tin có sẵn.

        Không cung cấp bất kỳ liên kết nguồn nào không có trong ngữ cảnh. Không tự tạo ra bất kỳ liên kết nguồn nào.

        Câu trả lời của Trợ lý Y tế:"""

        return prompt

    def generate_response(
            self,
            query: str,
            retrieved_docs: List[Dict[str, Any]],
            chat_history: Optional[List[Dict[str, str]]] = None,
        ) -> Dict[str, Any]:
    
        try:
           
            # Extract content from documents for context
            doc_texts = [doc["content"] for doc in retrieved_docs]
            
            # Combine retrieved documents into a single context
            context = "\n\n===DOCUMENT SECTION===\n\n".join(doc_texts)
            
            # Build the prompt
            prompt = self._build_prompt(query, context, chat_history)
            
            # Generate response
            response = self.response_generator_model.invoke(prompt)
            
            # Extract sources for citation
            sources = self._extract_sources(retrieved_docs) if hasattr(self, 'include_sources') and self.include_sources else []
            
            # Calculate confidence
            confidence = self._calculate_confidence(retrieved_docs)

            # Get response text based on type
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            safety_disclaimer = "\n\n⚠️ **Lưu ý quan trọng:** Thông tin trên chỉ mang tính chất tham khảo và được tạo ra bởi AI. Đây không phải là chẩn đoán y tế chính thức. Bạn nên đi khám bác sĩ chuyên khoa sớm nhất có thể để được thăm khám và điều trị phù hợp."
            
            if hasattr(self, 'include_sources') and self.include_sources and sources:
                response_with_source = response_text + safety_disclaimer + "\n\n##### Tài liệu nguồn:"
                for current_source in sources:
                    source_path = current_source['path']
                    source_title = current_source['title']
                    response_with_source += f"\n- [{source_title}]({source_path})"
            else:
                response_with_source = response_text + safety_disclaimer
            
            # Format final response - ensure we always return a string in the response key
            result = {
                "response": response_with_source,
                "sources": sources,
                "confidence": confidence
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
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
            
            # Skip if no source information is available
            if not source:
                continue
                
            # Create a unique identifier for this source
            source_id = f"{source}|{source_path}"
            
            # Skip if we've already included this source
            if source_id in seen_sources:
                continue
                
            # Add to our sources list
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

    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
    
        if not documents:
            return 0.0
            
        # Use combined score (both reranker and cosine similarity) if available, otherwise use original score
        if "combined_score" in documents[0]:
            scores = [doc.get("combined_score", 0) for doc in documents[:3]]
        elif "rerank_score" in documents[0]:
            scores = [doc.get("rerank_score", 0) for doc in documents[:3]]
        else:
            scores = [doc.get("score", 0) for doc in documents[:3]]
            
        # Average of top 3 document scores or fewer if less than 3
        return sum(scores) / len(scores) if scores else 0.0