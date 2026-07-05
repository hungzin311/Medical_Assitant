import requests
from typing import Dict

from .tavily_search import TavilySearchAgent

class WebSearchAgent:
    """
    Agent responsible for retrieving real-time medical information from web sources.
    """
    
    def __init__(self, config):
        self.tavily_search_agent = TavilySearchAgent()    
    def search(self, query: str) -> str:
        """
        Perform both general and medical-specific searches.
        """
        tavily_results = self.tavily_search_agent.search_tavily(query=query)
        return f"Tavily Results:\n{tavily_results}\n"
