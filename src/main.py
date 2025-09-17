import argparse
import os
from rag.retriever import DocumentRetriever
from rag.generator import AnswerGenerator

from config import VECTOR_DB_PATH, EMBEDDING_MODEL, LLM_MODEL, TOP_K

class RAGPipeline:
    def __init__(self):
        self.retriever = DocumentRetriever()
        self.generator = AnswerGenerator()

    def run(self, query: str):
        """RAG 파이프라인을 실행합니다."""
        print(f"질문: {query}")
        
        retrieved_docs = self.retriever.retrieve(query)
        print("\n[검색된 문서 조각]")
        for i, doc in enumerate(retrieved_docs):
            print(f"{i+1}. {doc[:100]}...")

        answer = self.generator.generate(question=query, context=retrieved_docs)
        print("\n[최종 답변]")
        print(answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="질문을 입력하세요.")
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    pipeline = RAGPipeline()
    pipeline.run(args.query)

    # 실행 예시: python main.py --query "이러닝시스템 사업의 요구사항을 정리해줘"