from dotenv import load_dotenv
import os

load_dotenv()

# 경로 설정
DATA_PATH = "./data/raw"
PDF_PATH = "./data/pdf"
VECTOR_DB_PATH = "src/vectorstore/vector_db"
VECTOR_STORE_PATH = "src/vectorstore/faiss_index"

# 모델 설정 (시나리오 B: OpenAI 기준)
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# RAG 파라미터
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 5