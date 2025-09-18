from config import Config
from dotenv import load_dotenv, find_dotenv
import fitz  # PyMuPDF
import os
import pandas as pd
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.schema import Document

def load_documents():
    def ext(original_filename, ext='pdf'):
        base_filename, _ = os.path.splitext(original_filename)
        return f"{base_filename}.{ext}"

    df = pd.read_csv(os.path.join(Config.DATA_PATH, "data_list.csv"))

    # NaN 값을 빈 문자열로 대체하여 메타데이터에 문제가 없도록 처리
    df = df.fillna('')

    doc_group = []

    for index, row in df.iterrows():
        # page_content는 '텍스트' 컬럼의 내용으로 설정
        print(f"이건 {index + 1}번째 문서: {row['사업명']}")
        metadata = {
            '공고 번호': row['공고 번호'],
            '사업명': row['사업명'],
            '사업 금액': row['사업 금액'],
            '발주 기관': row['발주 기관'],
            '공개 일자': row['공개 일자'],
            '입찰 참여 시작일': row['입찰 참여 시작일'],
            '입찰 참여 마감일': row['입찰 참여 마감일'],
            '사업 요약': row['사업 요약'],
            '파일명': row['파일명']
        }

        # Document 객체 생성
        docs = chunk(ext(row['파일명']), metadata=metadata)
        doc_group.append(docs)
        """
        try:
            docs = load_documents(ext(row['파일명']), metadata=metadata)
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading document {row['파일명']}: {e}")
            continue
        """
    print(f"총 {len(doc_group)}개의 문서가 로드되었습니다.")
    print(doc_group[0])
    return doc_group

def chunk(filename: str, metadata, page_chunk_size: int = 1500, page_chunk_overlap: int = 150, final_chunk_size: int = 500, final_chunk_overlap: int = 100) -> List[Document]:
    """
    1. TextSplitter를 이용해 페이지 단위로 텍스트를 1차 분할
    2. RecursiveCharacterTextSplitter를 이용해 1차 분할된 텍스트를 2차 분할
    """
    # PDF 파일에서 페이지 단위로 텍스트를 추출하는 함수
    def extract_pages_as_text(filename: str) -> List[str]:
        """
        PyMuPDF를 사용하여 PDF의 각 페이지 텍스트를 추출합니다.
        """
        doc = fitz.open(f"{Config.PDF_PATH}/{filename}")
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        doc.close()
        return pages_text

    # 1단계: 페이지 단위로 텍스트를 분할 (1차 분할)
    pages_text = extract_pages_as_text(filename)
    
    # TextSplitter 초기화 (여기서는 페이지를 하나의 덩어리로 간주)
    # 실제로는 TextSplitter 대신, 페이지를 그대로 사용하는 것이 더 자연스럽습니다.
    # 여기서는 "두 단계"를 보여주기 위해 TextSplitter의 기본 기능을 활용합니다.
    page_splitter = CharacterTextSplitter(chunk_size=page_chunk_size,chunk_overlap=page_chunk_overlap)
    first_stage_chunks = page_splitter.create_documents([t for t in pages_text if t.strip()])
    
    # 2단계: RecursiveCharacterTextSplitter를 이용해 재귀적으로 텍스트 분할 (2차 분할)
    # 1차 분할된 덩어리들을 다시 분할합니다.
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=final_chunk_size,chunk_overlap=final_chunk_overlap,separators=["\n\n", "\n", " ", ""])
    
    final_chunks = []
    for doc_chunk in first_stage_chunks:
        # 1차 청크의 내용을 다시 2차 청킹
        sub_chunks = recursive_splitter.create_documents([doc_chunk.page_content])
        
        # 메타데이터 추가 (원본 파일 정보)
        for sub_chunk in sub_chunks:
            sub_chunk.metadata = metadata
            final_chunks.append(sub_chunk)
            
    return final_chunks