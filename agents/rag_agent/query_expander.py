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
        self.logger.info(f"Expanding query")
        
        # Generate expansions using chat history if available
        expanded_query = self._generate_expansions(original_query, chat_history)
        
        return {
            "original_query": original_query,
            "expanded_query": expanded_query.content
        }
    
    def _generate_expansions(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Use LLM to expand query with medical terminology using chat history for context."""
        prompt = f"""
        Bạn là chuyên gia y tế giúp tạo ra các QUERY TÌM KIẾM mở rộng để tìm thông tin y tế chính xác.
        
        Câu hỏi gốc: {query}
        Lịch sử trò chuyện: {chat_history}
        
        NHIỆM VỤ: Tạo ra 2-3 câu truy vấn tìm kiếm (search queries) khác nhau để tìm thông tin y tế liên quan.
        
        HƯỚNG DẪN CHO MULTILINGUAL E5 EMBEDDING:
        - Tạo các câu truy vấn hoàn chỉnh, không phải câu hỏi phản hồi
        - Mỗi query nên tập trung vào một khía cạnh khác nhau của vấn đề
        - Sử dụng thuật ngữ y tế chính xác và từ đồng nghĩa
        - Bao gồm cả tiếng Việt và thuật ngữ y tế quốc tế
        - Tạo query cụ thể để tránh trùng lặp với các bệnh khác
        - Mỗi query trên một dòng riêng biệt
        - Nếu người dùng chỉ hỏi về các triệu chứng thì không đưa ra các câu hỏi liên quan đến cách chữa trị
        
        VÍ DỤ:
        Câu hỏi: "Tôi bị nổi mẩn đỏ và ngứa"
        Query mở rộng:
        triệu chứng nổi mẩn đỏ ngứa trên da nguyên nhân
        phát ban da đỏ kèm ngứa có thể là dấu hiệu của viêm da dị ứng dermatitis eczema
        các bệnh lý da gây nổi mẩn đỏ và ngứa như urticaria mề đay viêm da tiếp xúc
        
        Câu hỏi: "Thuốc điều trị cảm cúm"
        Query mở rộng:
        các loại thuốc điều trị cảm cúm cảm lạnh hiệu quả
        thuốc kháng virus antiviral oseltamivir điều trị influenza
        phương pháp điều trị triệu chứng cảm cúm hạ sốt giảm đau
        
        Câu hỏi: "Triệu chứng tiểu đường"
        Query mở rộng:
        triệu chứng bệnh tiểu đường type 1 type 2 dấu hiệu nhận biết
        diabetes mellitus biểu hiện lâm sàng khát nước tiểu nhiều
        chẩn đoán tiểu đường đái tháo đường xét nghiệm đường huyết
        
        Hãy tạo 2-3 query tìm kiếm cho câu hỏi trên (mỗi query một dòng, không đánh số):
        """
        expansion = self.model.invoke(prompt)
        
        return expansion