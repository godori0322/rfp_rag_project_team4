from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from typing import List
import os

from config import EMBEDDING_MODEL, VECTOR_DB_PATH

class VectorDBQuery:
    def __init__(self, db_path: str, embedding_model_name: str):
        # self.embeddings = OpenAIEmbeddings(model=embedding_model_name)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.db = FAISS.load_local(db_path, self.embeddings, allow_dangerous_deserialization=True)
        print(f"✅ Vector Store 불러오기 완료: {db_path}")

    def query_index(self, query: str, k: int = 5) -> List[str]:
        print(f"\n🔎 Querying for: '{query}'")
        results = self.db.similarity_search(query, k=k)
        
        return [doc.page_content for doc in results]

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    vector_db_query = VectorDBQuery(
        db_path=VECTOR_DB_PATH,
        embedding_model_name=EMBEDDING_MODEL
    )
    
    example_query = "이러닝 시스템의 주요 요구사항은 무엇인가요?"
    
    search_results = vector_db_query.query_index(example_query, k=1)
    
    print("\n[🔎 검색 결과]")
    for i, doc in enumerate(search_results):
        print(f"--- 문서 {i+1} ---\n{doc[:200]}...\n")
