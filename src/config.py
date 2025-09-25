from dotenv import load_dotenv
from dataclasses import dataclass
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

@dataclass
class Config:
    DATA_PATH = "./data/raw"
    PDF_PATH = "./data/pdf"
    VECTOR_DB_PATH = "./data/vectorstore"
    HISTORY_PATH = "./data/history"

    # 모델 설정 (시나리오 B: OpenAI 기준)
    EMBEDDING_MODEL = "text-embedding-3-small"
    # EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
    LLM_MODEL = "gpt-4o-mini"
    RFP_COLLECTION = 'rfp_documents'
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # RAG 파라미터
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    TOP_K = 5
    TEMPERATURE = 0.2

class LangSmithConfig:
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rfp_rag_project")
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")

    @classmethod
    def validate_config(cls):
        if not cls.LANGCHAIN_API_KEY:
            raise ValueError("LANGCHAIN_API_KEY is not set in environment variables")