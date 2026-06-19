from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
import os
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv
from httpx import Client
load_dotenv()

class OpenAIEmbeddings(Embeddings):
    def __init__(self, base_url: str = None, api_key: str = None, model_name: str = None):
        self.base_url = base_url or os.getenv("EMBEDDING_BASE_URL") 
        self.api_key = "empty"
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def embed_query(self, text: str) -> list[float]:
        if text is None:
            return []
        response = self.client.embeddings.create(input=text, model=self.model_name)
        return response.data[0].embedding
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        return [item.embedding for item in response.data]

    async def aembed_query(self, text: str) -> list[float]:
        if text is None: 
            return []
        response = await self.async_client.embeddings.create(
            input=text,
            model=self.model_name
        )
        return response.data[0].embedding 
    
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self.async_client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        return [item.embedding for item in response.data]

def get_gemini_llm(temperature=0.7):
    return get_llm(temperature=temperature)
def get_gemini_llm_2(temperature=0.7):
    return get_llm(temperature=temperature)

def get_gemini_llm_3(temperature=0.7):
    return get_llm(temperature=temperature)

def get_gemini_vision_llm(temperature=0.2):
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",  
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,
        convert_system_message_to_human=False,
        max_output_tokens=256,
        streaming=True,
    )

def get_polyp_vqa_llm(temperature=0.1):
    model = os.getenv("POLYP_VQA_MODEL")
    base_url = os.getenv("POLYP_VQA_BASE_URL")
    api_key = os.getenv("POLYP_VQA_API_KEY", "empty")

    if model and base_url:
        client = Client(verify=False)
        return ChatOpenAI(
            model=model,
            openai_api_base=base_url,
            openai_api_key=api_key,
            http_client=client,
            temperature=temperature,
            streaming=True,
        )

    return get_gemini_vision_llm(temperature=temperature)

def get_qwen_extra_body(max_completion_tokens=1024):
    return {
        "chat_template_kwargs": {"enable_thinking": False},
        "max_completion_tokens": max_completion_tokens,
    }

def get_graph_db():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    return Neo4jGraph(url=uri, username=user, password=password)

def get_embedding(): 
    return OpenAIEmbeddings(model_name="google/embeddinggemma-300m")

def get_llm(temperature=0.2):
    client = Client(verify=False)
    return ChatOpenAI(
        model="Qwen/Qwen3.6-27B",
        openai_api_base= os.getenv('LLM_BASE_URL'),
        openai_api_key="empty",
        http_client=client,
        streaming=True,
        temperature=temperature,
    )
