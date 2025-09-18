from dotenv import load_dotenv, find_dotenv
import fitz  # PyMuPDF
import os
import pandas as pd
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.schema import Document
from document import load_documents
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List

from config import Config

def create_vectorstore():
    load_dotenv(find_dotenv())

    # 환경 변수 가져오기
    my_var = os.getenv('OPENAI_API_KEY')
    print(my_var)
    doc_group = load_documents()
    embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL,openai_api_key=Config.OPENAI_API_KEY)
    dummy_docs = [Document(page_content="초기 생성을 위한 더미 문서입니다.")]
    vector_store = FAISS.from_documents(dummy_docs, embeddings)
    cnt = 1
    for doc in doc_group:
        vector_store.add_documents(doc)
        print(f"문서 추가 완료: 페이지 {cnt}")
        cnt += 1

    vector_store.save_local(Config.VECTOR_DB_PATH)

    print(f"Vector Store 저장 완료: {Config.VECTOR_DB_PATH}")

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    create_vectorstore()