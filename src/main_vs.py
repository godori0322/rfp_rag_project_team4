from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.schema import Document
from document import load_documents
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List

from config import Config

def create_vectorstore():
    load_dotenv(find_dotenv())

    # 환경 변수 가져오기
    if (len(Config.OPENAI_API_KEY) < 20):
        print('OPENAI_API_KEY 길이가 20보다 작습니다. 제대로 설정했는지 확인해 보겠어요?')
        return
    doc_group = load_documents()
    embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL,openai_api_key=Config.OPENAI_API_KEY)
    dummy_docs = [Document(page_content="초기 생성을 위한 더미 문서입니다.")]
    vector_store = Chroma.from_documents(documents=dummy_docs, embedding=embeddings, persist_directory=Config.VECTOR_DB_PATH, collection_name=Config.RFP_COLLECTION)

    cnt = 1
    for doc in doc_group:
        vector_store.add_documents(doc)
        print(f"문서 추가 완료: 페이지 {cnt}")
        cnt += 1

    vector_store.persist()
    print(f"저장된 문서 수: {vector_store._collection.count()}") # 저장된 문서 수 확인
    print(f"Vector Store 저장 완료: {Config.VECTOR_DB_PATH}")

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    create_vectorstore()
