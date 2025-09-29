from typing import List, Dict, Any, Optional
import json, re
class QueryExpander:
    """
    Expands user queries with medical terminology to improve retrieval.
    """
    def __init__(self, config):
        self.config = config
        self.model = config.rag.llm
        
    def expand_query(self, original_query: str, mode:str = "rag", patient_info:Dict =None, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        print("Starting Expanding query")
        # Generate expansions using chat history if available
        if mode == "rag":
            expanded_query = self._generate_expansions(original_query, chat_history)
            expanded_query = expanded_query.content
        elif mode == "kg":
            expanded_query = self._generate_kg_refine_query(original_query, patient_info, chat_history)

        return {
            "original_query": original_query,
            "expanded_query": expanded_query
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
    
    def _generate_kg_refine_query(self, query: str, patient_info: Dict = None, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        prompt = f"""
        Bạn là trợ lý y tế AI, nhiệm vụ của bạn là CHỈ chuẩn hoá lại truy vấn của người dùng
        để truy vấn Knowledge Graph và mô tả ngắn gọn thông tin bệnh nhân.
        Tuyệt đối KHÔNG bịa thêm triệu chứng hay bệnh.

        ────────────────────────────────
        DỮ LIỆU ĐẦU VÀO
        ────────────────────────────────
        THÔNG TIN Bệnh nhân (có thể thiếu): {patient_info}

        LỊCH SỬ TRÒ CHUYỆN (nếu có): {chat_history}

        CÂU HỎI/MIÊU TẢ GỐC CỦA NGƯỜI DÙNG:
        \"\"\"{query}\"\"\"

        ────────────────────────────────
        NHIỆM VỤ 1 – REFINE TRUY VẤN
        ────────────────────────────────
        1. Chỉ sử dụng thông tin người dùng đã nêu (kể cả trong lịch sử).
        2. KHÔNG tự bịa thêm bất cứ triệu chứng, bệnh hay ý định mới.
        3. BẢO TOÀN PHỦ ĐỊNH/LOẠI TRỪ: Nếu có cụm như "không", "không kèm", "không có",
           "chưa", "không bị" (ví dụ: "không đau ngực", "không có yếu tố X"), PHẢI giữ NGUYÊN
           trong truy vấn đã chuẩn hoá (không được lược bỏ hay suy diễn ngược).
        4. TỔNG QUÁT HÓA THÔNG TIN Y TẾ: Có thể khái quát hóa các thông tin cụ thể thành 
           thuật ngữ y tế chung (vd. "huyết áp cao kéo dài 1 tuần" → "huyết áp cao",
           "đau đầu 3 ngày" → "đau đầu").
        5. Nếu người dùng liệt kê TRIỆU CHỨNG → chuẩn hoá thành một TRUY VẤN MÔ TẢ TRUNG LẬP
           (không bắt buộc ở dạng câu hỏi), tập trung vào truy xuất bệnh liên quan.
           Ví dụ có thể dùng:
           - "các bệnh có triệu chứng A, B, …; không có/không kèm C, D"
           - "triệu chứng: A, B; không kèm C, D"
           KHÔNG tự động chuyển thành câu hỏi "Đây là bệnh gì…".
        6. Nếu người dùng hỏi về BỆNH hoặc ĐIỀU TRỊ cụ thể:
           – Khái quát tên bệnh khi cần (vd. "tiểu đường loại 2" → "tiểu đường"),
           – Giữ nguyên mục tiêu (triệu chứng, điều trị, nguyên nhân…).
        7. Giữ nguyên phong cách của người dùng (hỏi/miêu tả/mệnh lệnh) nếu phù hợp; không đổi mục đích.
        8. Câu văn phải đầy đủ, tự nhiên bằng tiếng Việt.

        ────────────────────────────────
        NHIỆM VỤ 2 – MÔ TẢ BỆNH NHÂN
        ────────────────────────────────
        Tóm tắt thành 1–2 câu ngắn gọn, chỉ dùng thông tin có thật:
        - Tuổi / giới tính (nếu biết),
        - Tiền sử bệnh, tình trạng hiện tại, triệu chứng chính (bao gồm cả phủ định quan trọng nếu có).

        ────────────────────────────────
        ĐỊNH DẠNG KẾT QUẢ
        ────────────────────────────────
        CHỈ TRẢ VỀ JSON, KHÔNG THÊM BẤT KỲ KÝ TỰ NÀO KHÁC.
        Trả về đúng JSON sau (KHÔNG thêm trường khác):

        {{
        "refined_question": "<truy vấn đã chuẩn hoá (không bắt buộc dạng câu hỏi; giữ phủ định nếu có)>",
        "patient_context": "<mô tả bệnh nhân>"
        }}
        """

        response = self.model.invoke(prompt)
        raw = response.content.strip() 
        match = re.search(r"\{[\s\S]*?\}", raw)
        parsed = json.loads(match.group(0))
        # print("Parsed:", parsed)

        return{ 
            'refined_question': parsed['refined_question'],
            'patient_context': parsed['patient_context']
        }
        