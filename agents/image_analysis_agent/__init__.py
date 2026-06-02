from .image_classifier import ImageClassifier
from .general_diagnosis_agent import GeneralMedicalDiagnosisAgent
from .summarizer_agent import MedicalImageSummarizer
from .polyp_seg_tool.polyp_seg_inference import PolypSegmentation
from .polyp_vqa_agent import PolypVQAAgent
import os
import uuid

class ImageAnalysisAgent:
    """
    Agent responsible for processing image uploads and classifying them as medical or non-medical, and determining their type.
    """
    
    def __init__(self, config):
        self.image_classifier = ImageClassifier(vision_model=config.medical_cv.llm)
        self.general_diagnosis_agent = GeneralMedicalDiagnosisAgent(vision_model=config.medical_cv.llm)
        self.polyp_vqa_agent = PolypVQAAgent(vision_model=config.medical_cv.polyp_vqa_llm)
        self.summarizer = MedicalImageSummarizer(llm=config.medical_cv.summarizer_llm)
        self.polyp_seg_agent = PolypSegmentation(model_path=config.medical_cv.polyp_seg_model_path)
        self.polyp_seg_output_path = config.medical_cv.polyp_seg_output_path
        self.polyp_seg_output_dir = config.medical_cv.polyp_seg_output_dir
    
    # classify image
    def analyze_image(self, image_path: str, user_query: str = None) -> str:
        return self.image_classifier.classify_image(image_path, user_query)
    #polyp segmentation
    def segment_polyp(self, image_path: str, output_path: str = None) -> str:
        if output_path is None:
            os.makedirs(self.polyp_seg_output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(
                self.polyp_seg_output_dir,
                f"{base_name}_{uuid.uuid4().hex[:8]}_seg.jpg",
            )
        return self.polyp_seg_agent.predict(image_path, output_path)
    # polyp visual question answering
    def answer_polyp_vqa(self, image_path: str, segmentation_image_path: str, user_query: str = None) -> dict:
        return self.polyp_vqa_agent.answer_question(image_path, segmentation_image_path, user_query)
    # general diagnosis
    def diagnose_general_medical_image(self, image_path: str, user_query: str = None) -> dict:
        return self.general_diagnosis_agent.diagnose_image(image_path, user_query)
    #summarizer
    def summarize_diagnosis(self, diagnosis_result: dict, chat_history=None, user_query=None) -> dict:
        return self.summarizer.summarize_diagnosis(diagnosis_result, chat_history, user_query)
    
    def generate_followup_response(self, image_id: str, follow_up_query: str) -> str:
        return self.summarizer.generate_followup_response(image_id, follow_up_query)
