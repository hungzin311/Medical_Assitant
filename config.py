"""
Configuration file for the Multi-Agent Medical Chatbot

This file contains all the configuration parameters for the project.

If you want to change the LLM and Embedding model:

you can do it by changing all 'llm' and 'embedding_model' variables present in multiple classes below.

Each llm definition has unique temperature value relevant to the specific class. 
"""

import os
from dotenv import load_dotenv
from llm_config import get_gemini_llm, get_gemini_vision_llm, get_together_embeddings

# Load environment variables from .env file
load_dotenv()

class AgentDecisoinConfig:
    def __init__(self):
        self.llm = get_gemini_llm(temperature=0.1)  # Deterministic

class ConversationConfig:
    def __init__(self):
        self.llm = get_gemini_llm(temperature=0.7)  # Creative but factual

class WebSearchConfig:
    def __init__(self):
        self.llm = get_gemini_llm(temperature=0.3)  # Slightly creative but factual
        self.context_limit = 20     # include last 20 messsages (10 Q&A pairs) in history

class RAGConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"
        self.embedding_dim = 1024  # Together AI m2-bert embedding dimension
        self.distance_metric = "Cosine"
        # Set use_local to False to use cloud Qdrant
        self.use_local = False
        self.vector_local_path = "./data/qdrant_vietnamese_db"
        self.doc_local_path = "./data/docs_vietnamese_db"
        self.parsed_content_dir = "./data/parsed_docs_vietnamese_db"
        # Make sure these environment variables are set in your .env file
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = "medical_assistance_rag_vietnamese"
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.embedding_model = get_together_embeddings()
        self.llm = get_gemini_llm(temperature=0.1)  # Slightly creative but factual
        self.summarizer_model = get_gemini_llm(temperature=0.2)  # Slightly creative but factual
        self.chunker_model = get_gemini_llm(temperature=0.0)  # factual
        self.response_generator_model = get_gemini_llm(temperature=0.2)  # Slightly creative but factual
        self.top_k = 10
        self.vector_search_type = 'similarity'  # or 'mmr'

        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = 5

        self.max_context_length = 8192

        self.include_sources = True

        self.min_retrieval_confidence = 0.80

        self.context_limit = 20     # include last 20 messsages (10 Q&A pairs) in history

class MedicalCVConfig:
    def __init__(self):
        self.brain_tumor_model_path = "./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_segmentation.pth"
        self.chest_xray_model_path = "./agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth"
        self.skin_lesion_model_path = "./agents/image_analysis_agent/skin_lesion_agent/models/checkpointN25_.pth.tar"
        self.skin_lesion_segmentation_output_path = "./uploads/skin_lesion_output/segmentation_plot.png"
        self.llm = get_gemini_vision_llm(temperature=0.1)  # Use vision-specific model for medical image analysis
        self.summarizer_llm = get_gemini_llm(temperature=0.3)  # Slightly creative for summarization
        self.polyp_seg_model_path = "./agents/image_analysis_agent/polyp_seg_tool/models/deeplabv3_resnet50.pth"
        self.polyp_seg_output_path = "./uploads/polyp_seg_output/polyp_seg_image_output.jpg"

class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "SKIN_LESION_AGENT": True,
            "POLYP_SEGMENTATION_AGENT": True,
            "GENERAL_MEDICAL_IMAGE_AGENT": True
        }
        self.validation_timeout = 300
        self.default_action = "reject"

class APIConfig:
    def __init__(self):
        self.host = "127.0.0.1"  # or "localhost"
        self.port = 3000
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 5  # MB

class UIConfig:
    def __init__(self):
        self.theme = "light"
        self.enable_speech = True
        self.enable_image_upload = True

class Config:
    def __init__(self):
        self.agent_decision = AgentDecisoinConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_conversation_history = 20  # Include last 20 messsages (10 Q&A pairs) in history

# # Example usage
# config = Config()