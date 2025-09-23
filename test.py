from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from httpx import Client

client = Client(verify=False)

llm = ChatOpenAI(
    model="google/medgemma-27b-text-it",
    openai_api_base="https://my-container-gmzsmq3d-8000.serverless.fptcloud.com/v1",
    openai_api_key="your_api_key",
    http_client=client,  
)

resp = llm.invoke("Hãy giải thích lợi ích của MRI trong chẩn đoán thần kinh")
print(resp.content)
