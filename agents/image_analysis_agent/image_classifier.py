import os
import json
import base64
from mimetypes import guess_type

from typing import TypedDict
from langchain_core.output_parsers import JsonOutputParser

class ClassificationDecision(TypedDict):
    """Output structure for the decision agent."""
    image_type: str
    reasoning: str
    confidence: float

class ImageClassifier:
    """Uses GPT-4o Vision to analyze images and determine their type."""
    
    def __init__(self, vision_model):
        self.vision_model = vision_model
        self.json_parser = JsonOutputParser(pydantic_object=ClassificationDecision)
        
    def local_image_to_data_url(self, image_path: str) -> str:
        """
        Get the url of a local image
        """
        mime_type, _ = guess_type(image_path)

        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

        return f"data:{mime_type};base64,{base64_encoded_data}"
    
    def classify_image(self, image_path: str, user_query: str = None) -> str:
        """Analyzes the image to classify it as a medical image and determine it's type."""
        print(f"[ImageAnalyzer] Analyzing image: {image_path}")

        vision_prompt = [
            {"role": "user", "content": [
                {"type": "text", "text": (
                    f"""
                    System: You are an expert in medical imaging. Analyze the uploaded image.
                    
                    {f'CÂU HỎI CỦA NGƯỜI DÙNG: {user_query}' if user_query else ''}
                    
                    Phân tích hình ảnh này và xác định loại:
                    
                    **Các loại hình ảnh y tế:**
                    - 'POLYP SEGMENTATION': Hình ảnh nội soi đại tràng có polyp (colonoscopy images with polyps)
                    - 'SKIN LESION SEGMENTATION': Hình ảnh tổn thương da (skin lesion images)  
                    - 'GENERAL MEDICAL IMAGE': Hình ảnh y tế khác (X-ray, CT, MRI, etc.)
                    - 'NON-MEDICAL': Không phải hình ảnh y tế
                    
                    **Hướng dẫn phân loại:**
                    - Nếu là ảnh nội soi đại tràng (màu hồng/đỏ, có cấu trúc ruột, có thể có polyp) → 'POLYP SEGMENTATION'
                    - Nếu là ảnh da với tổn thương/nốt ruồi/vết bất thường → 'SKIN LESION SEGMENTATION'
                    - Nếu là ảnh y tế khác (X-ray, siêu âm, CT, MRI, etc.) → 'GENERAL MEDICAL IMAGE'
                    - Nếu không phải ảnh y tế → 'NON-MEDICAL'
                    
                    Trả lời theo định dạng JSON:
                    {{
                    "image_type": "LOẠI HÌNH ẢNH",
                    "reasoning": "Lý do phân loại từng bước",
                    "confidence": 0.95
                    }}
                    """
                )},
                {"type": "image_url", "image_url": {"url": self.local_image_to_data_url(image_path)}}
            ]}
        ]
        
        # Invoke LLM to classify the image
        response = self.vision_model.invoke(vision_prompt)

        try:
            # Ensure the response is parsed as JSON
            response_json = self.json_parser.parse(response.content)
            return response_json  # Returns a dictionary instead of a string
        except json.JSONDecodeError:
            print("[ImageAnalyzer] Warning: Response was not valid JSON.")
            return {"image_type": "unknown", "reasoning": "Invalid JSON response", "confidence": 0.0}

        # return response.content.strip().lower()