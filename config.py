from dotenv import load_dotenv
import os

load_dotenv()

# API CREDENTIALS
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# LLM MODEL (configurable via .env, defaults to flash-lite)
model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash-lite")
