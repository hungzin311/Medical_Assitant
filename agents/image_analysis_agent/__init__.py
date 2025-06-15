from .image_classifier import ImageClassifier
from .skin_lesion_agent.skin_lesion_inference import SkinLesionSegmentation
from .general_diagnosis_agent import GeneralMedicalDiagnosisAgent
from .summarizer_agent import MedicalImageSummarizer

class ImageAnalysisAgent:
    """
    Agent responsible for processing image uploads and classifying them as medical or non-medical, and determining their type.
    """
    
    def __init__(self, config):
        self.image_classifier = ImageClassifier(vision_model=config.medical_cv.llm)
        self.skin_lesion_agent = SkinLesionSegmentation(model_path=config.medical_cv.skin_lesion_model_path)
        self.skin_lesion_segmentation_output_path = config.medical_cv.skin_lesion_segmentation_output_path
        self.general_diagnosis_agent = GeneralMedicalDiagnosisAgent(vision_model=config.medical_cv.llm)
        self.summarizer = MedicalImageSummarizer(llm=config.medical_cv.summarizer_llm)
    
    # classify image
    def analyze_image(self, image_path: str, user_query: str = None) -> str:
        """Classifies images as medical or non-medical and determines their type."""
        return self.image_classifier.classify_image(image_path, user_query)
    
    def segment_skin_lesion(self, image_path: str) -> str:
        return self.skin_lesion_agent.predict(image_path, self.skin_lesion_segmentation_output_path)
        
    def diagnose_general_medical_image(self, image_path: str, user_query: str = None) -> dict:
        """Diagnoses general medical images using MLLM."""
        return self.general_diagnosis_agent.diagnose_image(image_path, user_query)
        
    def summarize_diagnosis(self, diagnosis_result: dict, chat_history=None, user_query=None) -> dict:
        """Summarizes diagnosis results and provides follow-up recommendations."""
        return self.summarizer.summarize_diagnosis(diagnosis_result, chat_history, user_query)
        
    def generate_followup_response(self, image_id: str, follow_up_query: str) -> str:
        """Generates a response to a follow-up query based on stored diagnosis."""
        return self.summarizer.generate_followup_response(image_id, follow_up_query)
