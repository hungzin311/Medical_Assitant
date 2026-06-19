import os
from .web_search_agent import WebSearchAgent
from typing import Dict, List, Optional
from dotenv import load_dotenv
from utils.streaming import invoke_with_streaming

load_dotenv()

class WebSearchProcessor:
    def __init__(self, config):
        self.web_search_agent = WebSearchAgent(config)
        
        # Initialize LLM for processing web search results
        self.llm = config.web_search.llm
    
    def _build_prompt_for_web_search(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:

        prompt = f"""Đây là những tin nhắn gần đây từ cuộc trò chuyện của chúng ta:

        {chat_history}

        Người dùng đã hỏi câu hỏi sau:

        {query}

        Tóm tắt chúng thành một câu hỏi đầy đủ, được hình thành tốt chỉ khi cuộc trò chuyện trước đó có vẻ liên quan đến câu hỏi hiện tại để có thể sử dụng cho tìm kiếm web.
        Hãy giữ nó ngắn gọn và đảm bảo nó nắm bắt được ý định chính đằng sau cuộc thảo luận.
        """

        return prompt
    
    def process_web_results(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        
        web_search_query_prompt = self._build_prompt_for_web_search(query=query, chat_history=chat_history)
        from utils.llm_config import get_qwen_extra_body

        web_search_query = self.llm.bind(extra_body=get_qwen_extra_body()).invoke(web_search_query_prompt)
        
        # Retrieve web search results
        web_results = self.web_search_agent.search(web_search_query.content)     

        llm_prompt = (
            "Bạn là một trợ lý AI chuyên về thông tin y tế. Dưới đây là kết quả tìm kiếm web "
            "được truy xuất cho câu hỏi của người dùng. Hãy tóm tắt và tạo ra một câu trả lời hữu ích, ngắn gọn. "
            "Chỉ sử dụng các nguồn đáng tin cậy và đảm bảo độ chính xác về y tế.\n\n"
            f"Câu hỏi: {query}\n\nKết quả tìm kiếm web:\n{web_results}\n\nCâu trả lời:"
        )
        
        # Invoke the LLM to process the results
        response = invoke_with_streaming(self.llm, llm_prompt)
        
        # Add safety disclaimer to web search response
        safety_disclaimer = "\n\n⚠️ **Lưu ý quan trọng:** Thông tin trên chỉ mang tính chất tham khảo và được tạo ra bởi AI. Đây không phải là chẩn đoán y tế chính thức. Bạn nên đi khám bác sĩ chuyên khoa sớm nhất có thể để được thăm khám và điều trị phù hợp."
        
        # Get response content and add disclaimer
        if hasattr(response, 'content'):
            response_text = response.content + safety_disclaimer
        else:
            response_text = str(response) + safety_disclaimer
        
        return response_text
