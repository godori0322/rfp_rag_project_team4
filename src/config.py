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
    # RERANK_MODEL = 'BAAI/bge-reranker-base'
    RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2' #평가용 입니다.
    RFP_COLLECTION = 'rfp_documents'
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    IS_LOCAL = True

    # RAG 파라미터
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    FIRST_TOP_K = 7
    TOP_K = 3
    FETCH_K = 20
    LAMBDA_MULT = 0.5
    TEMPERATURE = 0.2

    @classmethod
    def to_local(cls):
        cls.EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
        cls.LLM_MODEL = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
        cls.VECTOR_DB_PATH = "./data/vectorstore_local"

class LangSmithConfig:
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rfp_rag_project")
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")

    @classmethod
    def validate_config(cls):
        if not cls.LANGCHAIN_API_KEY:
            raise ValueError("LANGCHAIN_API_KEY is not set in environment variables")