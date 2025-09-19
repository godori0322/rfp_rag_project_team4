from dotenv import load_dotenv
from dataclasses import dataclass
import os

@dataclass
class Config:
    DATA_PATH = "./data/raw"
    PDF_PATH = "./data/pdf"
    VECTOR_DB_PATH = "./data/vectorstore/vector_db"

    # 모델 설정 (시나리오 B: OpenAI 기준)
    EMBEDDING_MODEL = "text-embedding-3-small"
    # EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
    LLM_MODEL = "gpt-4o-mini"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # RAG 파라미터
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    TOP_K = 5
    TEMPERATURE = 0.2
