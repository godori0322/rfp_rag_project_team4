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

    if not Config.OPENAI_API_KEY or len(Config.OPENAI_API_KEY) < 20:
        print("OPENAI_API_KEY가 올바르게 설정되지 않았습니다. .env 파일을 확인해주세요.")
        return
    
    print("문서 로딩을 시작합니다...")
    doc_group = load_documents()
    print(f"총 {len(doc_group)}개의 문서 그룹을 로드했습니다.")

    embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL, openai_api_key=Config.OPENAI_API_KEY)
    
    vector_store = Chroma(
        persist_directory=Config.VECTOR_DB_PATH, 
        embedding_function=embeddings, 
        collection_name=Config.RFP_COLLECTION
    )

    print("\n기존 DB의 내용을 초기화합니다...")
    vector_store.delete_collection()
    # 다시 컬렉션을 생성
    vector_store = Chroma(
        persist_directory=Config.VECTOR_DB_PATH, 
        embedding_function=embeddings, 
        collection_name=Config.RFP_COLLECTION
    )


    print("\n벡터 DB에 문서 추가를 시작합니다...")
    cnt = 1
    for doc_chunks in doc_group:
        # None 메타데이터를 방지하는 코드
        for chunk in doc_chunks:
            if chunk.metadata is None:
                chunk.metadata = {}
        
        vector_store.add_documents(doc_chunks)
        print(f"{cnt}번 문서 그룹 추가 완료. (청크 갯수: {len(doc_chunks)})")
        cnt += 1

    print("\n모든 문서의 벡터 변환 및 저장이 완료되었습니다.")
    print(f"저장된 총 문서(청크) 수: {vector_store._collection.count()}")
    print(f"Vector Store 경로: {Config.VECTOR_DB_PATH}")

if __name__ == "__main__":
    create_vectorstore()

