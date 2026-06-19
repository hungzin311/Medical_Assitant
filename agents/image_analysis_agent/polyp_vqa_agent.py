import base64
import logging
from mimetypes import guess_type
from typing import Any, Dict, Optional


class PolypVQAAgent:
    """
    Agent for answering visual questions about colonoscopy polyp images.
    It sends both the original image and the segmentation overlay to a VLM.
    """

    def __init__(self, vision_model):
        self.logger = logging.getLogger(__name__)
        self.vision_model = vision_model
        self.logger.info("Polyp VQA Agent initialized")

    def local_image_to_data_url(self, image_path: str) -> str:
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

        return f"data:{mime_type};base64,{base64_encoded_data}"

    def answer_question(
        self,
        image_path: str,
        segmentation_image_path: str,
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        question = (question or "").strip()
        if not question:
            question = "Hãy mô tả các phát hiện chính liên quan đến polyp trong ảnh nội soi này."

        prompt_text = f"""
        Bạn là chuyên gia nội soi đại trực tràng và thị giác y tế, chuyên trả lời VQA về polyp.

        Input gồm:
        - Ảnh 1: ảnh nội soi polyp gốc.
        - Ảnh 2: ảnh đã được segment/overlay từ mô hình polyp segmentation.
          Quy ước màu segment nếu có:
          - Vùng đỏ: polyp tân sinh (neoplastic), nguy cơ cao hơn.
          - Vùng xanh lá: polyp không tân sinh (non-neoplastic), thường nguy cơ thấp hơn.

        Câu hỏi của người dùng:
        {question}

        Yêu cầu trả lời:
        1. Trả lời trực tiếp câu hỏi dựa trên cả ảnh gốc và ảnh segment.
        2. Nêu rõ quan sát hình ảnh liên quan đến câu trả lời.
        3. Nếu câu hỏi yêu cầu phân loại/nguy cơ, hãy giải thích mức độ chắc chắn dựa trên bằng chứng thị giác.
        4. Nếu ảnh segment không rõ hoặc mâu thuẫn với ảnh gốc, hãy nói rõ giới hạn đó.
        5. Nhắc ngắn gọn rằng đây là hỗ trợ AI, không thay thế kết luận của bác sĩ nội soi/giải phẫu bệnh.

        Trả lời bằng tiếng Việt, súc tích và có cấu trúc.
        """

        vision_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": self.local_image_to_data_url(image_path)},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": self.local_image_to_data_url(segmentation_image_path)},
                    },
                ],
            }
        ]

        try:
            from utils.llm_config import get_qwen_extra_body

            response = self.vision_model.bind(extra_body=get_qwen_extra_body()).invoke(vision_prompt)
            return {
                "answer": response.content,
                "success": True,
                "image_path": image_path,
                "segmentation_image_path": segmentation_image_path,
            }
        except Exception as e:
            self.logger.error(f"Error answering polyp VQA: {e}")
            return {
                "answer": "Tôi đã gặp lỗi khi trả lời câu hỏi VQA về polyp. Vui lòng thử lại hoặc tham khảo bác sĩ chuyên khoa.",
                "success": False,
                "error": str(e),
                "image_path": image_path,
                "segmentation_image_path": segmentation_image_path,
            }
