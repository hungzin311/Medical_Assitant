import logging
from typing import List, Dict, Any, Optional

class QueryExpander:
    """
    Expands user queries with medical terminology to improve retrieval.
    """
    def __init__(self, config):
        self.logger = logging.getLogger(f"{self.__module__}")
        self.config = config
        self.model = config.rag.llm
        
    def expand_query(self, original_query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Expand the original query with relevant medical terms.
        
        Args:
            original_query: The user's original query
            chat_history: Optional chat history for context
            
        Returns:
            Dictionary with original and expanded queries
        """
        self.logger.info(f"Expanding query: {original_query}")
        
        # Generate expansions using chat history if available
        expanded_query = self._generate_expansions(original_query, chat_history)
        
        return {
            "original_query": original_query,
            "expanded_query": expanded_query.content
        }
    
    def _generate_expansions(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Use LLM to expand query with medical terminology using chat history for context."""
        prompt = f"""
        Với vai trò là một chuyên gia y tế, hãy mở rộng câu hỏi sau đây với thuật ngữ y tế liên quan, 
        các từ đồng nghĩa và các khái niệm liên quan sẽ giúp truy xuất thông tin y tế phù hợp:
        
        {chat_history}Câu hỏi người dùng: {query}
        
        Chỉ mở rộng câu hỏi nếu bạn cảm thấy cần thiết, nếu không hãy giữ nguyên câu hỏi của người dùng.
        Hãy cụ thể với lĩnh vực y tế hoặc bất kỳ lĩnh vực nào khác được đề cập trong câu hỏi của người dùng, không thêm các lĩnh vực y tế khác. 
        Không tự mở rộng câu hỏi với các ý khác mà người dùng không hỏi.
        Hãy đưa ra kết quả mở rộng ngắn gọn và dễ hiểu. 
        Chỉ cung cấp câu hỏi mở rộng mà không có giải thích.
        
        Dựa vào lịch sử cuộc trò chuyện và câu hỏi hiện tại, hãy mở rộng câu hỏi để bao gồm các khái niệm y tế liên quan đã được đề cập trong cuộc trò chuyện trước đó.
        """
        expansion = self.model.invoke(prompt)
        
        return expansion