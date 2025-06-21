import os
from .web_search_agent import WebSearchAgent
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

class WebSearchProcessor:
    """
    Processes web search results and routes them to the appropriate LLM for response generation.
    """
    
    def __init__(self, config):
        self.web_search_agent = WebSearchAgent(config)
        
        # Initialize LLM for processing web search results
        self.llm = config.web_search.llm
    
    def _build_prompt_for_web_search(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Build the prompt for the web search.
        
        Args:
            query: User query
            chat_history: chat history
            
        Returns:
            Complete prompt string
        """
        # Add chat history if provided
        # print("Chat History:", chat_history)
            
        # Build the prompt
        prompt = f"""Đây là những tin nhắn gần đây từ cuộc trò chuyện của chúng ta:

        {chat_history}

        Người dùng đã hỏi câu hỏi sau:

        {query}

        Tóm tắt chúng thành một câu hỏi đầy đủ, được hình thành tốt chỉ khi cuộc trò chuyện trước đó có vẻ liên quan đến câu hỏi hiện tại để có thể sử dụng cho tìm kiếm web.
        Hãy giữ nó ngắn gọn và đảm bảo nó nắm bắt được ý định chính đằng sau cuộc thảo luận.
        """

        return prompt
    
    def process_web_results(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Fetches web search results, processes them using LLM, and returns a user-friendly response.
        """
        # print(f"[WebSearchProcessor] Fetching web search results for: {query}")
        web_search_query_prompt = self._build_prompt_for_web_search(query=query, chat_history=chat_history)
        # print("Web Search Query Prompt:", web_search_query_prompt)
        web_search_query = self.llm.invoke(web_search_query_prompt)
        # print("Web Search Query:", web_search_query)
        
        # Retrieve web search results
        web_results = self.web_search_agent.search(web_search_query.content)

        # print(f"[WebSearchProcessor] Fetched results: {web_results}")
        
        # Construct prompt to LLM for processing the results
        llm_prompt = (
            "Bạn là một trợ lý AI chuyên về thông tin y tế. Dưới đây là kết quả tìm kiếm web "
            "được truy xuất cho câu hỏi của người dùng. Hãy tóm tắt và tạo ra một câu trả lời hữu ích, ngắn gọn. "
            "Chỉ sử dụng các nguồn đáng tin cậy và đảm bảo độ chính xác về y tế.\n\n"
            f"Câu hỏi: {query}\n\nKết quả tìm kiếm web:\n{web_results}\n\nCâu trả lời:"
        )
        
        # Invoke the LLM to process the results
        response = self.llm.invoke(llm_prompt)
        
        # Add safety disclaimer to web search response
        safety_disclaimer = "\n\n⚠️ **Lưu ý quan trọng:** Thông tin trên chỉ mang tính chất tham khảo và được tạo ra bởi AI. Đây không phải là chẩn đoán y tế chính thức. Bạn nên đi khám bác sĩ chuyên khoa sớm nhất có thể để được thăm khám và điều trị phù hợp."
        
        # Get response content and add disclaimer
        if hasattr(response, 'content'):
            response_text = response.content + safety_disclaimer
        else:
            response_text = str(response) + safety_disclaimer
        
        return response_text
