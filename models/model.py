import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from config import model_name, GEMINI_API_KEY

# Semaphore to cap concurrent LLM calls and avoid hitting API quotas.
# This limits total in-flight Gemini requests across all agents.
LLM_SEMAPHORE = asyncio.Semaphore(10)


class ThrottledChat(ChatGoogleGenerativeAI):
    """Wraps all async generation calls with a semaphore."""

    async def _agenerate(self, *args, **kwargs):
        async with LLM_SEMAPHORE:
            return await super()._agenerate(*args, **kwargs)


chat = ThrottledChat(
    model=model_name,
    api_key=GEMINI_API_KEY,
    max_retries=10,
    timeout=120,
)
