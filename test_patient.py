from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from logging import getLogger
from proxy_setting import set_proxy

set_proxy()

load_dotenv()

def main():
    print('hello abcd') 
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Gemini 2.0 supports multimodal inputs
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1,
        convert_system_message_to_human=False,
        max_output_tokens=4096  # Increase token limit for detailed image analysis
    )
    logger = getLogger(__name__)
    logger.info('Starting the application')
    res = llm.invoke('hello abcd')
    print(res)
    logger.info('Application finished')

if __name__ == "__main__": 
    main()