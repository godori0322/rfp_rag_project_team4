import os
from rag.retriever import DocumentRetriever
from rag.generator import AnswerGenerator

class RAGPipeline:
    def __init__(self):
        self.retriever = DocumentRetriever()
        self.generator = AnswerGenerator()

    def run(self, query: str):
        print(f"질문: {query}")
        
        retrieved_docs = self.retriever.retrieve(query)
        print("\n[검색된 문서 청크]")
        for i, doc in enumerate(retrieved_docs):
            print(f"{i+1}. {doc[:100]}...")

        answer = self.generator.generate(question=query, context=retrieved_docs)
        print("\n[최종 답변]")
        print(answer)

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    pipeline = RAGPipeline()
    sample_query = "사무실 네트워크의 요구 사항을 알려줘."
    pipeline.run(sample_query)