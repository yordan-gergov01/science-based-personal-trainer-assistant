import os
from pathlib import Path

try:
    import streamlit as st
    # Running on Streamlit Cloud
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
    IS_STREAMLIT_CLOUD = True
except:
    # Running locally
    from dotenv import load_dotenv
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    IS_STREAMLIT_CLOUD = False

BASE_DIR = Path(__file__).parent.parent
PDF_DIRECTORY = str(BASE_DIR / "PDFs")
VECTOR_DB_PATH = str(BASE_DIR / "faiss_index")
LOGS_PATH = str(BASE_DIR / "logs")

os.makedirs(LOGS_PATH, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile" 
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

RETRIEVAL_K = 6
SEARCH_TYPE = "similarity"

TEMPERATURE = 0.1
MAX_TOKENS = 1000

MODULE_CATEGORIES = {
    'nutrition': [
        'ad libitum', 'adherence', 'biochemistry', 'carbohydrates', 
        'dietary fat', 'energy', 'fasting', 'ketogenic', 'macronutrition',
        'micronutrition', 'nutrition case', 'periodization', 'protein',
        'supplements', 'health science and food'
    ],
    'training': [
        'advanced strength', 'age specific', 'cardio', 'exercise library',
        'exercise performance', 'exercise selection', 'how to structure',
        'injury management', 'powerlifting', 'program customization',
        'stretching', 'training case', 'training gear', 'training volume',
        'understanding muscle', 'warming up', 'posture'
    ],
    'science': [
        'biochemistry', 'muscle functional anatomy', 'understanding muscle growth'
    ],
    'lifestyle': [
        'business', 'fitness for women', 'lifestyle factors', 
        'how to learn think and research'
    ]
}

LOG_LEVEL = "INFO"

SYSTEM_NAME = "RAG Coach"
SYSTEM_VERSION = "1.0.0"