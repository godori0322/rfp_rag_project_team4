import os
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from rag.retriever import DocumentRetriever
from rag.generator import AnswerGenerator

from config import Config

# RAG 체인 파이프라인 구현
# class RAGPipeline:
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

# VECTOR_DB_PATH에서 벡터 DB 로드, 이후 RAG 체인 파이프라인 실행
class RAGPipeline:
    def __init__():
        self.retriever = load_vectorstore().as_retriever(search_type="similarity", search_kwargs={"k": Config.TOP_K})
        self.llm = ChatOpenAI(model_name=Config.LLM_MODEL, temperature=Config.TEMPERATURE)

        # ✅ 1. Prompt 정의
        self.prompt = ChatPromptTemplate.from_template("""
        당신은 RFP(제안요청서) 분석 전문가입니다. 
        주어진 문서를 바탕으로 사용자의 질문에 대해 한국어로 답변해주세요.
        문서에 내용이 없으면 '문서에서 관련 정보를 찾을 수 없습니다.'라고 답변하세요.
        
        Context:
        {context}
        
        Question:
        {question}
        """)

        # ✅ 2. Chain 정의 (LCEL 방식)
        self.chain = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    # Retriever가 반환한 문서들을 문자열로 합치기
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 체인 실행
    def run(self, query: str):
        return self.chain.invoke(query)

def load_vectorstore():
    vectorstore = FAISS.load_local(Config.VECTOR_DB_PATH, embeddings=Config.EMBEDDINGS, allow_dangling_deserialization=True)
    return vectorstore

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    pipeline = RAGPipeline()
    sample_query = "사무실 네트워크의 요구 사항을 알려줘."
    pipeline.run(sample_query)