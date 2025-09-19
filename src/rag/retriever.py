import os
from typing import List

from vectorstore.query_index import VectorDBQuery
from config import VECTOR_DB_PATH, EMBEDDING_MODEL, TOP_K

class DocumentRetriever:
    def __init__(self):
        self.db_query = VectorDBQuery(
            db_path=VECTOR_DB_PATH,
            embedding_model_name=EMBEDDING_MODEL
        )
        self.k = TOP_K
        print("✅ DocumentRetriever 생성.")

    def retrieve(self, query: str) -> List[str]:
        print(f"🔄 Retrieving documents for query: '{query}'")
        retrieved_docs = self.db_query.query_index(query, k=self.k)
        return retrieved_docs

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    retriever = DocumentRetriever()
    
    example_query = "연례 행사의 목표를 알려줘."
    
    search_results = retriever.retrieve(example_query)
    
    print("\n[🔍 검색 결과]")
    for i, doc in enumerate(search_results):
        print(f"--- 문서 {i+1} ---\n{doc[:200]}...\n")