from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_together import TogetherEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def get_gemini_llm(temperature=0.7):
    """Initialize and return a Gemini Pro LLM instance"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,
        convert_system_message_to_human=False
    )

def get_gemini_vision_llm(temperature=0.2):
    """Initialize and return a Gemini Pro Vision LLM instance for image analysis"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Gemini 2.0 supports multimodal inputs
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,
        convert_system_message_to_human=False,
        max_output_tokens=4096  # Increase token limit for detailed image analysis
    )

def get_together_embeddings():
    """Initialize and return Together AI embeddings instance"""
    return TogetherEmbeddings(
        model="intfloat/multilingual-e5-large-instruct",
        together_api_key=os.getenv("TOGETHER_API_KEY")
    ) 