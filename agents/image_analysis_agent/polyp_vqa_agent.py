import base64
import logging
from mimetypes import guess_type
from typing import Any, Dict, Optional


class PolypVQAAgent:
    """
    Agent for answering visual questions about colonoscopy polyp images.
    It supports both original-image VQA and segmentation-grounded VQA.
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

    def _normalize_question(self, question: Optional[str]) -> str:
        question = (question or "").strip()
        if not question:
            question = "What are the main findings in this endoscopic image?"
        return question

    def _invoke_vision_model(
        self,
        vision_prompt: list,
        image_path: str,
        segmentation_mask_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            from utils.llm_config import get_qwen_extra_body

            response = self.vision_model.bind(extra_body=get_qwen_extra_body()).invoke(vision_prompt)
            result = {
                "answer": response.content,
                "success": True,
                "image_path": image_path,
            }
            if segmentation_mask_path:
                result["segmentation_image_path"] = segmentation_mask_path
                result["segmentation_mask_path"] = segmentation_mask_path
            return result
        except Exception as e:
            self.logger.error(f"Error answering polyp VQA: {e}")
            result = {
                "answer": "Tôi đã gặp lỗi khi trả lời câu hỏi VQA về polyp. Vui lòng thử lại hoặc tham khảo bác sĩ chuyên khoa.",
                "success": False,
                "error": str(e),
                "image_path": image_path,
            }
            if segmentation_mask_path:
                result["segmentation_image_path"] = segmentation_mask_path
                result["segmentation_mask_path"] = segmentation_mask_path
            return result

    def answer_question_with_segmentation_mask(
        self,
        original_image_path: str,
        segmentation_mask_path: str,
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        question = self._normalize_question(question)

        prompt_text = f"""Original image:
Segmentation mask:
Using both the original image and segmentation mask, {question}
Answer:"""

        vision_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": self.local_image_to_data_url(original_image_path)},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": self.local_image_to_data_url(segmentation_mask_path)},
                    },
                ],
            }
        ]

        return self._invoke_vision_model(
            vision_prompt,
            image_path=original_image_path,
            segmentation_mask_path=segmentation_mask_path,
        )

    def answer_question_original_image_only(
        self,
        original_image_path: str,
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        question = self._normalize_question(question)

        prompt_text = f"""Original image:
{question}
Answer:"""

        vision_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": self.local_image_to_data_url(original_image_path)},
                    },
                ],
            }
        ]

        return self._invoke_vision_model(vision_prompt, image_path=original_image_path)

    def answer_question(
        self,
        image_path: str,
        segmentation_image_path: Optional[str] = None,
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        if segmentation_image_path:
            return self.answer_question_with_segmentation_mask(
                original_image_path=image_path,
                segmentation_mask_path=segmentation_image_path,
                question=question,
            )
        return self.answer_question_original_image_only(
            original_image_path=image_path,
            question=question,
        )
