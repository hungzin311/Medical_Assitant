import os
import threading
from dotenv import load_dotenv
from utils.llm_config import get_embedding, get_gemini_llm, get_gemini_vision_llm, get_polyp_vqa_llm

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
        self.context_limit = 20     
class RAGConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"
        self.embedding_dim = 768  
        self.distance_metric = "Cosine"
        self.use_local = False
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = "medical_assistance_rag_vietnamese"
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.embedding_model = get_embedding()
        self.llm = get_gemini_llm(temperature=0.0)  
        self.top_k = 10
        self.vector_search_type = 'similarity'  # or 'mmr'

        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = 5

        self.max_context_length = 8192

        self.include_sources = True

        self.min_retrieval_confidence = 0.70

        self.context_limit = 20     


class MedlinePlusConfig:
    def __init__(self):
        self.collection_name = os.getenv("MEDLINEPLUS_COLLECTION_NAME", "medlineplus_kb")
        self.data_dir = os.getenv(
            "MEDLINEPLUS_DATA_DIR",
            "/home/hung/Coding/VectorDB/data/medlineplus",
        )
        self.embedding_model = get_embedding()
        self.llm = get_gemini_llm(temperature=0.0)
        self.top_k = int(os.getenv("MEDLINEPLUS_TOP_K", "8"))
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.vector_name = os.getenv("MEDLINEPLUS_VECTOR_NAME", "dense")
        self.include_relations = os.getenv("MEDLINEPLUS_INCLUDE_RELATIONS", "true").lower() == "true"
        self.max_relations = int(os.getenv("MEDLINEPLUS_MAX_RELATIONS", "20"))
        self.min_retrieval_confidence = float(os.getenv("MEDLINEPLUS_MIN_RETRIEVAL_CONFIDENCE", "0.70"))
        self.context_limit = int(os.getenv("MEDLINEPLUS_CONTEXT_LIMIT", "20"))


class PatientDBConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"
        self.embedding_dim = 768  
        self.distance_metric = "Cosine"
        self.use_local = False
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = "medical_records"  
        self.embedding_model = get_embedding()
        self.top_k = 50  
        self.llm = get_gemini_llm(temperature=0.1) 
        self.max_age_range_default = 10  
        self.min_cases_for_confidence = 5  
        self.outbreak_threshold = 5  
        self.risk_score_threshold = 0.7  
        
        # Population monitoring settings
        self.monitoring_time_windows = ["7d", "30d", "90d"]
        self.alert_confidence_threshold = 0.6
        self.max_population_results = 1000

class MedicalCVConfig:
    def __init__(self):
        self.llm = get_gemini_vision_llm(temperature=0.1) 
        self.polyp_vqa_llm = get_polyp_vqa_llm(temperature=0.1)
        self.summarizer_llm = get_gemini_llm(temperature=0.3) 
        self.polyp_seg_model_path = "./agents/image_analysis_agent/polyp_seg_tool/models/deeplabv3_resnet50.pth"
        self.polyp_seg_output_path = "./uploads/polyp_seg_output/polyp_seg_image_output.jpg"
        self.polyp_seg_output_dir = "./uploads/polyp_seg_output"

class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "MEDLINEPLUS_AGENT": False,
            "POLYP_SEGMENTATION_AGENT": True,
            "POLYP_VQA_AGENT": True,
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
    _instance = None 
    _lock = threading.Lock()
    def __new__(cls): 
        if cls._instance is None: 
            with cls._lock: 
                if cls._instance is None: 
                    cls._instance = super(Config, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.agent_decision = AgentDecisoinConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medlineplus = MedlinePlusConfig()
        self.patient_db = PatientDBConfig()  # Add patient database configuration
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_conversation_history = 20  # Include last 20 messsages (10 Q&A pairs) in history
        print("Config initialized successfully")
        
        self._initialized = True
