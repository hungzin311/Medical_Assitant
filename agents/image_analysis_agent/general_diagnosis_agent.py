import os
import logging
from typing import Dict, Any, Optional
import base64
from mimetypes import guess_type
class GeneralMedicalDiagnosisAgent:
    """
    Agent responsible for diagnosing general medical images using MLLM.
    This agent handles images classified as "OTHER" by the image classifier.
    """
    
    def __init__(self, vision_model):
        """
        Initialize the general medical diagnosis agent.
        
        Args:
            vision_model: The multimodal LLM model for image analysis
        """
        self.logger = logging.getLogger(__name__)
        self.vision_model = vision_model
        self.logger.info("General Medical Diagnosis Agent initialized")
    
    def local_image_to_data_url(self, image_path: str) -> str:
        """
        Convert local image to data URL for MLLM input.
        
        Args:
            image_path: Path to the local image
            
        Returns:
            Data URL of the image
        """
        
        
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "application/octet-stream"
            
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")
            
        return f"data:{mime_type};base64,{base64_encoded_data}"
    
    def diagnose_image(self, image_path: str, user_query: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a medical image using MLLM and provide a diagnosis.
        
        Args:
            image_path: Path to the medical image
            user_query: Optional user query for context
            
        Returns:
            Dictionary containing diagnosis results
        """
        self.logger.info(f"Analyzing medical image: {image_path}")
        
        # Prepare the prompt for the vision model
        if user_query:
            prompt_text = f"""
            Analyze this medical image in detail. The user has asked: "{user_query}"
            
            Provide a comprehensive analysis with the following structure:
            1. Description of what you see in the image
            2. Possible medical conditions or abnormalities
            3. Recommended next steps or further tests
            
            Important: Clearly state that this is an AI analysis and not a replacement for professional medical diagnosis.
            """
        else:
            prompt_text = """
            Analyze this medical image in detail.
            
            Provide a comprehensive analysis with the following structure:
            1. Description of what you see in the image
            2. Possible medical conditions or abnormalities
            3. Recommended next steps or further tests
            
            Important: Clearly state that this is an AI analysis and not a replacement for professional medical diagnosis.
            """
        
        # Create vision prompt
        vision_prompt = [
            {"role": "system", "content": "You are a medical imaging expert. Analyze the uploaded medical image carefully and provide detailed insights."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": self.local_image_to_data_url(image_path)}}
            ]}
        ]
        
        try:
            # Invoke MLLM for analysis
            response = self.vision_model.invoke(vision_prompt)
            
            return {
                "diagnosis": response.content,
                "success": True,
                "image_path": image_path
            }
        except Exception as e:
            self.logger.error(f"Error analyzing medical image: {e}")
            return {
                "diagnosis": "I encountered an error while analyzing this medical image. Please try again or consult a healthcare professional.",
                "success": False,
                "error": str(e),
                "image_path": image_path
            } 