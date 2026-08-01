from config import TAVILY_API_KEY
from tavily import AsyncTavilyClient

client = AsyncTavilyClient(
    api_key=TAVILY_API_KEY
)