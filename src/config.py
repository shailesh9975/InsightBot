# src/config.py
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# --- OpenAI API Key ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Model Settings ---
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
LLM_MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.7 # For LLM creativity

# --- RAG Parameters ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 5 # Number of relevant chunks to retrieve

# --- Streamlit UI Settings ---
APP_TITLE = "InsightBot: Your Personalized Knowledge Assistant"
