import os
import json
import base64
from mimetypes import guess_type

from typing import TypedDict
from langchain_core.output_parsers import JsonOutputParser

class ClassificationDecision(TypedDict):
    """Output structure for the decision agent."""
    image_type: str
    eng_query: str
    confidence: float

class ImageClassifier:
    """Uses a vision model to classify medical images and refine the user query."""

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
                    You are a medical image classifier.

                    User query: {user_query or ""}

                    Return ONLY valid JSON with:
                    - image_type: one of "POLYP SEGMENTATION", "GENERAL MEDICAL IMAGE", "NON-MEDICAL"
                    - eng_query: a short English VQA question for the image. If the user query is empty, use "What are the main findings in this endoscopic image?"
                    - confidence: a number from 0 to 1

                    Classify colonoscopy/endoscopic images with possible polyps as "POLYP SEGMENTATION".

                    JSON format:
                    {{
                      "image_type": "POLYP SEGMENTATION",
                      "eng_query": "What are the main findings in this endoscopic image?",
                      "confidence": 0.95
                    }}
                    """
                )},
                {"type": "image_url", "image_url": {"url": self.local_image_to_data_url(image_path)}}
            ]}
        ]
        
        # Invoke LLM to classify the image
        from utils.llm_config import get_qwen_extra_body

        response = self.vision_model.bind(extra_body=get_qwen_extra_body()).invoke(vision_prompt)

        try:
            # Ensure the response is parsed as JSON
            response_json = self.json_parser.parse(response.content)
            return response_json  # Returns a dictionary instead of a string
        except Exception:
            print("[ImageAnalyzer] Warning: Response was not valid JSON.")
            return {
                "image_type": "unknown",
                "eng_query": user_query or "What are the main findings in this image?",
                "confidence": 0.0,
            }

        # return response.content.strip().lower()
