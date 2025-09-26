from config import Config
#from dotenv import load_dotenv, find_dotenv
#import fitz  # PyMuPDF
import pdfplumber
import os
import pandas as pd
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.schema import Document
import string

def load_documents():
    def ext(original_filename, ext='pdf'):
        base_filename, _ = os.path.splitext(original_filename)
        return f"{base_filename}.{ext}"

    df = pd.read_csv(os.path.join(Config.DATA_PATH, "data_list.csv"))

    # 날짜 컬럼들을 datetime 타입으로 표준화.
    # errors='coerce'는 변환 불가능한 값을 NaT(Not a Time)으로 처리.
    date_columns = ['공개 일자', '입찰 참여 시작일', '입찰 참여 마감일']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # NaN 값을 빈 문자열로 대체하여 다른 메타데이터에 문제가 없도록 처리.
    df = df.fillna('')
    doc_group = []
    annotations = []

    for index, row in df.iterrows():
        # 표준화된 datetime 객체를 "YYYY-MM-DD" 형식의 문자열로 변환.
        # 날짜 정보가 없는(NaT) 경우는 빈 문자열 ''로 처리.
        publish_date_str = row['공개 일자'].strftime('%Y-%m-%d') if pd.notna(row['공개 일자']) else ''
        bid_start_date_str = row['입찰 참여 시작일'].strftime('%Y-%m-%d') if pd.notna(row['입찰 참여 시작일']) else ''
        bid_end_date_str = row['입찰 참여 마감일'].strftime('%Y-%m-%d') if pd.notna(row['입찰 참여 마감일']) else ''
        
        metadata = {
            'rfp_number': row['공고 번호'],
            'project_title': row['사업명'],
            'budget_krw': row['사업 금액'],
            'agency': row['발주 기관'],
            
            # 최종 변환된 날짜 문자열을 메타데이터에 할당.
            'publish_date': publish_date_str,
            'bid_start_date': bid_start_date_str,
            'bid_end_date': bid_end_date_str,

            'summary': row['사업 요약'],
            'filename': row['파일명']
        }

        # Document 객체 생성
        docs = chunk(os.path.join(Config.PDF_PATH, ext(row['파일명'])), metadata=metadata)
        doc_group.append(docs)

        annotations.append(f"이건 {index + 1}번째 문서. 총 청크갯수: {len(docs)}. {row['파일명']}")
        print(annotations[len(annotations) - 1])
        
    print(f"총 {len(doc_group)}개의 문서가 로드되었습니다.")
    for anno in annotations:
        print(anno)
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
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                # extra_attrs로 'size'를 가져오도록 설정
                # extract_words -> 각 단어의 텍스트, 위치, 폰트 크기(size)

                words = page.extract_words(extra_attrs=["size", "fontname"])
                reconstructed_lines.extend(reconstruct_lines_from_words(words))
    except Exception as e:
        print(f"'{filepath}' 파일을 처리하는 중 오류가 발생했습니다: {e}")
        return [] # 오류 발생 시 빈 리스트 반환


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
        
        # if '목 차' == (" ".join(header.split()).strip()):
        #     print(f'### 다음은 목차내용이라 제외합니다 ((({content})))')
        #     continue

        # 내용이 긴 챕터만 다시 분할
        sub_chunks = recursive_splitter.split_text(content)
        for sub_chunk_content in sub_chunks:
            if is_high_special_char_ratio(sub_chunk_content):
                continue

            final_metadata = metadata.copy()
            final_metadata['parent_header'] = header
            doc = Document(page_content=sub_chunk_content, metadata=final_metadata)
            final_documents.append(doc)
        
    return final_documents


def is_high_special_char_ratio(text: str, threshold: float = 0.6) -> bool:
    SPECIAL_CHARS = set(string.punctuation + '`~!@#$%^&*()_+-=[]{}|;:",./<>?·') # 특수문자 정의 (구두점 및 기타 기호)
    
    total_length = len(text)
    if total_length == 0:
        return False

    special_char_count = sum(1 for char in text if char in SPECIAL_CHARS)
    special_char_ratio = special_char_count / total_length

    if special_char_ratio > threshold:
        print(f"---- 청크 제외됨 (특수문자 비율 {special_char_ratio:.2f}) ----")
        # print(f"제외된 청크 내용: {text}") # 너무 길어서 주석 처리
    return special_char_ratio > threshold

