import argparse
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from config import Config

def query(query: str):
    embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(Config.VECTOR_DB_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": Config.TOP_K})
    llm = ChatOpenAI(model_name=Config.LLM_MODEL, temperature=Config.TEMPERATURE)
    chain = create_chain(retriever, llm)
    return chain.invoke(query)

def create_chain(retriever, llm):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template("""
        당신은 RFP(제안요청서) 분석 전문가입니다. 
        주어진 문서를 바탕으로 사용자의 질문에 대해 한국어로 답변해주세요.
        문서에 내용이 없으면 '문서에서 관련 정보를 찾을 수 없습니다.'라고 답변하세요.
        
        Context:
        {context}
        
        Question:
        {question}
        """)
    chain = (
        {"context": retriever | format_docs,"question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    return chain

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="질문을 입력하세요.")
    args = parser.parse_args()

    response = query(args.query)
    print("\n[최종 답변]")
    print(f"\n{response}")

    # 실행 예시: python src/main.py --query "한국전력공사 RFP 문서의 주요 내용은 무엇인가요?"