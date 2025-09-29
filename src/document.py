# document.py

import os
import re
import pandas as pd
import pdfplumber
import numpy as np
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import Config

# --- Helper Functions ---

def convert_table_to_markdown(table: List[List[str]]) -> str:
    """PDFplumber로 추출된 테이블(list of lists)을 Markdown 형식으로 변환합니다."""
    markdown_table = ""
    if not table:
        return ""

    # 헤더 생성
    header = [str(cell) if cell is not None else "" for cell in table[0]]
    markdown_table += "| " + " | ".join(header) + " |\n"
    
    # 구분선 생성
    markdown_table += "| " + " | ".join(["---"] * len(header)) + " |\n"
    
    # 본문 생성
    for row in table[1:]:
        body = [str(cell) if cell is not None else "" for cell in row]
        markdown_table += "| " + " | ".join(body) + " |\n"
        
    return markdown_table

def clean_text_with_regex(text: str, patterns: List[str]) -> str:
    """주어진 정규 표현식 패턴들을 사용하여 텍스트를 청소합니다."""
    for pattern in patterns:
        text = re.sub(pattern, "", text)
    return text

# --- Main Functions ---

def load_documents():
    """CSV 메타데이터와 PDF 문서를 로드하고 청킹하여 Document 객체 리스트를 반환합니다."""
    def ext(original_filename, ext='pdf'):
        base_filename, _ = os.path.splitext(original_filename)
        return f"{base_filename}.{ext}"

    df = pd.read_csv(os.path.join(Config.DATA_PATH, "data_list.csv"))
    
    date_columns = ['공개 일자', '입찰 참여 시작일', '입찰 참여 마감일']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y%m%d').fillna(0).astype(int)

    df = df.fillna('')

    all_docs = []
    for index, row in df.iterrows():
        print(f"[{index + 1}/{len(df)}] 문서 처리 중: {row['사업명']}")
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
        
        filepath = os.path.join(Config.PDF_PATH, ext(row['파일명']))
        
        # 💡 개선된 chunk 함수 호출
        # 여러 파라미터를 실험해볼 수 있습니다.
        docs = chunk(
            filepath=filepath, 
            metadata=metadata,
            header_percentile=95, # 상위 1% 폰트 크기를 헤더로 간주
            final_chunk_size=2500, # 청크 사이즈 실험
            final_chunk_overlap=200  # 청크 오버랩 실험
        )
        all_docs.append(docs)

    print(f"총 {len(df)}개의 원본 문서에서 {len(all_docs)}개의 청크를 생성했습니다.")
    return all_docs


def chunk(filepath: str, 
          metadata: dict, 
          header_percentile: int = 95, 
          final_chunk_size: int = 2500, 
          final_chunk_overlap: int = 200,
          noise_patterns: List[str] = None
         ) -> List[Document]:
    """
    개선된 문서 처리 및 청킹 함수.

    1. (노이즈 제거) 정규 표현식으로 머리글/바닥글 제거
    2. (테이블 처리) 테이블을 Markdown으로 변환
    3. (동적 헤더 탐지) 폰트 크기 분포를 분석하여 동적으로 헤더 임계값 설정
    4. 1차 청킹: 헤더를 기준으로 의미 단위의 '챕터' 생성
    5. 2차 청킹: RecursiveCharacterTextSplitter로 '챕터'를 최종 크기로 분할
    """
    if noise_patterns is None:
        # 💡 일반적인 머리글/바닥글, 페이지 번호 제거 패턴 (필요시 추가/수정)
        noise_patterns = [
            r"^\s*-\s*\d+\s*-\s*$",  # "- 1 -", "- 2 -" போன்ற வடிவங்கள்
            r"^\s*\d+\s*$",         # 페이지 번호만 있는 경우
            r"(?i)page\s*\d+\s*of\s*\d+", # "Page 1 of 10"
        ]

    page_items = [] # 페이지의 텍스트와 테이블을 위치 정보와 함께 저장
    all_font_sizes = []

    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages):
            
            # --- 1. 테이블 추출 및 변환 ---
            tables = page.extract_tables()
            for table in tables:
                md_table = convert_table_to_markdown(table)
                # 테이블의 y 위치를 기준으로 저장 (나중에 텍스트와 순서대로 합치기 위함)
                table_bbox = page.find_tables()[0].bbox
                page_items.append({'type': 'table', 'content': md_table, 'top': table_bbox[1], 'page': page_num})

            # --- 2. 텍스트 추출 및 노이즈 제거 ---
            # 테이블 영역을 제외하고 텍스트 추출
            content_without_tables = page.filter(lambda obj: obj["object_type"] == "char")
            
            # 폰트 사이즈 수집 및 줄 단위 텍스트 재구성
            current_line = ""
            current_top = -1
            line_size = 10 # 기본 폰트 크기

            for char in content_without_tables.chars:
                if current_top != char['top']:
                    if current_line.strip():
                        cleaned_line = clean_text_with_regex(current_line, noise_patterns)
                        if cleaned_line.strip():
                            page_items.append({'type': 'text', 'content': cleaned_line, 'size': line_size, 'top': current_top, 'page': page_num})
                    
                    current_line = ""
                    current_top = char['top']
                    line_size = char.get('size', 10)
                
                current_line += char['text']
                all_font_sizes.append(char.get('size', 10))
            
            # 마지막 줄 추가
            if current_line.strip():
                cleaned_line = clean_text_with_regex(current_line, noise_patterns)
                if cleaned_line.strip():
                    page_items.append({'type': 'text', 'content': cleaned_line, 'size': line_size, 'top': current_top, 'page': page_num})


    # --- 3. 동적 헤더 임계값 계산 ---
    try:
        header_font_threshold = np.percentile(all_font_sizes, header_percentile)
    except IndexError: # 문서에 텍스트가 거의 없는 경우
        header_font_threshold = 18 # 기본값으로 대체

    # --- 4. 1차 청킹 (헤더 기준) ---
    # 페이지 아이템들을 페이지 번호와 수직 위치(top) 기준으로 정렬
    page_items.sort(key=lambda x: (x['page'], x['top']))
    
    font_size_chunks = []
    current_chunk_content = ""
    current_chunk_header = f"문서 시작 ({os.path.basename(filepath)})"

    for item in page_items:
        if item['type'] == 'text':
            font_size = round(item.get('size', 0))
            text = item['content']
            
            if font_size >= header_font_threshold:
                if current_chunk_content.strip():
                    font_size_chunks.append({"header": current_chunk_header, "content": current_chunk_content.strip()})
                current_chunk_header = text
                current_chunk_content = ""
            else:
                current_chunk_content += text + "\n"
        
        elif item['type'] == 'table':
            current_chunk_content += "\n" + item['content'] + "\n"

    if current_chunk_content.strip():
        font_size_chunks.append({"header": current_chunk_header, "content": current_chunk_content.strip()})

    # --- 5. 2차 청킹 (사이즈 기준) ---
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=final_chunk_size,
        chunk_overlap=final_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    final_documents = []
    for chapter in font_size_chunks:
        header = chapter['header']
        content = chapter['content']
        
        sub_chunks = recursive_splitter.split_text(content)
        for sub_chunk_content in sub_chunks:
            final_metadata = metadata.copy()
            final_metadata['parent_header'] = header
            doc = Document(page_content=sub_chunk_content, metadata=final_metadata)
            final_documents.append(doc)
            
    return final_documents