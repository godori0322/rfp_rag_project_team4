from config import Config
#from dotenv import load_dotenv, find_dotenv
#import fitz  # PyMuPDF
import pdfplumber
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
        # if index > 3:
        #     break
        metadata = {
            'rfp_number': row['공고 번호'],
            'project_title': row['사업명'],
            'budget_krw': row['사업 금액'],
            'agency': row['발주 기관'],
            'publish_date': row['공개 일자'],
            'bid_start_date': row['입찰 참여 시작일'],
            'bid_end_date': row['입찰 참여 마감일'],
            'summary': row['사업 요약'],
            'filename': row['파일명']
        }

        # Document 객체 생성
        docs = chunk(os.path.join(Config.PDF_PATH, ext(row['파일명'])), metadata=metadata)
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


def chunk(filepath: str, metadata: dict, header_font_threshold: int = 18, final_chunk_size: int = 500, final_chunk_overlap: int = 50) -> List[Document]:

    """
    header_font_threshold: int = 18, 
    --> 🐹 : 개인적으로 테스트 해봤을때 가장 좋았던 임계값으로 적용해놨습니다.
    
    ### 1차 수정
    1. 폰트 크기를 기준으로 구조적인 '챕터'를 먼저 나눕니다.
    2. 내용이 긴 '챕터'는 RecursiveCharacterTextSplitter로 다시 작게 분할합니다.
    
    ### 2차 수정
    1. pdfplumber를 사용해 단어 단위로 상세 정보 추출 (텍스트 추출)
    2. 폰트 크기로 임계값(Threshold) 기준으로 1차 청킹 (챕터 생성)
    3. RecursiveCharacterTextSplitter로 2차 청킹
    
    
    """
    # --- 로컬 헬퍼 함수 정의 ---
    def reconstruct_lines_from_words(words: List[dict[str, any]]) -> List[dict[str, any]]:
        """pdfplumber의 단어(word) 목록을 줄(line) 단위로 재구성합니다."""
        lines = []
        if not words:
            return []
        
        current_line_words = [words[0]]
        for i in range(1, len(words)):
            # 같은 줄에 있는지 확인 (수직 위치가 거의 동일한 경우)
            if abs(words[i]['top'] - words[i-1]['top']) < 2:
                current_line_words.append(words[i])
            else:
                # 새 줄 시작
                lines.append({
                    'text': ' '.join(w['text'] for w in current_line_words),
                    'size': current_line_words[0].get('size', 0) # 첫 단어의 크기를 대표로
                })
                current_line_words = [words[i]]
        
        # 마지막 줄 추가
        lines.append({
            'text': ' '.join(w['text'] for w in current_line_words),
            'size': current_line_words[0].get('size', 0)
        })
        return lines
    # --------------------------
    
    # pdfplumber를 사용하여 단어 단위로 상세 정보 추출
    reconstructed_lines = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            # extra_attrs로 'size'를 가져오도록 설정
            # extract_words -> 각 단어의 텍스트, 위치, 폰트 크기(size)
            
            words = page.extract_words(extra_attrs=["size", "fontname"])
            reconstructed_lines.extend(reconstruct_lines_from_words(words))

    # 폰트 크기를 기준으로 1차 청킹 (챕터 생성)
    font_size_chunks = []
    current_chunk_content = ""
    current_chunk_header = f"문서 시작 ({os.path.basename(filepath)})"

    for line in reconstructed_lines:
        font_size = round(line['size'])
        text = line['text']

        if font_size >= header_font_threshold:
            if current_chunk_content.strip():
                font_size_chunks.append({"header": current_chunk_header, "content": current_chunk_content.strip()})
            current_chunk_header = text
            current_chunk_content = ""
        else:
            current_chunk_content += text + "\n"
            
    if current_chunk_content.strip():
        font_size_chunks.append({"header": current_chunk_header, "content": current_chunk_content.strip()})

    # RecursiveCharacterTextSplitter를 사용하여 2차 청킹
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=final_chunk_size,
        chunk_overlap=final_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    final_documents = []
    for chapter in font_size_chunks:
        header = chapter['header']
        content = chapter['content']
        
        # 내용이 긴 챕터만 다시 분할
        sub_chunks = recursive_splitter.split_text(content)
        for sub_chunk_content in sub_chunks:
            final_metadata = metadata.copy()
            final_metadata['parent_header'] = header
            doc = Document(page_content=sub_chunk_content, metadata=final_metadata)
            final_documents.append(doc)
        
    return final_documents