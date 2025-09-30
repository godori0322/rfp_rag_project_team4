# document.py

import os
import re
import pandas as pd
import pdfplumber
import numpy as np
from typing import List, Dict, Any
import string
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import Config


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
        # Normalize whitespace: strip lines, replace multiple spaces/tabs, reduce newlines
        # 1. Strip leading/trailing spaces from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        # 2. Replace multiple spaces/tabs with a single space
        text = re.sub(r'[ \t]+', ' ', text)
        # 3. Reduce multiple newlines to a single newline
        text = re.sub(r'\n{2,}', '\n', text)
    return text


def load_documents():
    """CSV 메타데이터와 PDF 문서를 로드하고 청킹하여 Document 객체 리스트를 반환."""
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
        docs = chunk(
            filepath=filepath, 
            metadata=metadata,
            header_percentile=90, # 상위 10% 폰트 크기를 헤더로 간주
            final_chunk_size=1000, # 청크 사이즈 실험
            final_chunk_overlap=120  # 청크 오버랩 실험
        )
        all_docs.append(docs)

    print(f"총 {len(df)}개의 원본 문서에서 {len(all_docs)}개의 청크를 생성했습니다.")
    return all_docs


def chunk(filepath: str, 
          metadata: dict, 
          header_percentile: int = 90, 
          final_chunk_size: int = 1000, 
          final_chunk_overlap: int = 120,
          noise_patterns: List[str] = None
         ) -> List[Document]:
    """
    개선된 문서 처리 및 청킹 함수 (테이블 중복 제거 포함).

    1. (노이즈 제거) 정규 표현식으로 머리글/바닥글 제거
    2. (테이블 처리) 테이블을 Markdown으로 변환 (중복 방지)
    3. (텍스트 추출) extract_text() 기본 사용, 테이블 영역 제외
    4. (chars 기반 fallback) 테이블 영역 제외
    5. (동적 헤더 탐지) 폰트 크기 분포 분석
    6. 1차 청킹: 헤더 기준 챕터 생성
    7. 2차 청킹: RecursiveCharacterTextSplitter로 분할
    """
    if noise_patterns is None:
        noise_patterns = [
            r"^\s*-\s*\d+\s*-\s*$",      # "- 1 -", "- 2 -" (line only)
            r"-\s*\d+\s*-",                # "- 18 -" anywhere in line
            r"^\s*\d+\s*$",                # 페이지 번호만 있는 경우
            r"(?i)page\s*\d+\s*of\s*\d+" # "Page 1 of 10"
        ]

    page_items = []
    all_font_sizes = []

    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages):
            
            # --- 1. 테이블 추출 ---
            tables = page.find_tables()
            table_bboxes = [t.bbox for t in tables]  # 테이블 영역 bbox 저장
            for table in tables:
                md_table = convert_table_to_markdown(table.extract())
                page_items.append({
                    'type': 'table',
                    'content': md_table,
                    'top': table.bbox[1],
                    'page': page_num
                })

            # --- 2. 텍스트 추출 (테이블 영역 제외) ---
            text = page.extract_text(x_tolerance=5, y_tolerance=5)
            if text:
                lines = text.split('\n')
                filtered_lines = []

                # 라인별 bbox 정보가 없으므로 fallback용 chars로 y 위치 확인
                words = page.extract_words()
                line_tops = {}
                for w in words:
                    line_text = w['text']
                    top = round(w['top'])
                    line_tops.setdefault(top, []).append(line_text)

                for top, words_in_line in line_tops.items():
                    # 테이블 bbox와 겹치지 않으면 포함
                    if not any(b[1] <= top <= b[3] for b in table_bboxes):
                        line_text = " ".join(words_in_line)
                        filtered_lines.append(line_text)

                # Apply noise cleansing to each line individually
                cleaned_lines = [clean_text_with_regex(line, noise_patterns) for line in filtered_lines]
                cleaned_text = "\n".join([line for line in cleaned_lines if line.strip()])
                if cleaned_text.strip():
                    page_items.append({
                        'type': 'text',
                        'content': cleaned_text,
                        'size': 10,
                        'top': 0,
                        'page': page_num
                    })
            else:
                # --- 3. chars 기반 fallback (테이블 영역 제외) ---
                chars = page.chars
                non_table_chars = [c for c in chars if not any(
                    b[0] <= c["x0"] <= b[2] and b[1] <= c["top"] <= b[3] for b in table_bboxes
                )]

                tolerance = 5
                current_line, current_top, line_size = "", -1000, 10
                for char in non_table_chars:
                    if abs(current_top - char['top']) > tolerance:
                        if current_line.strip():
                            cleaned_line = clean_text_with_regex(current_line, noise_patterns)
                            if cleaned_line.strip():
                                page_items.append({
                                    'type': 'text',
                                    'content': cleaned_line,
                                    'size': line_size,
                                    'top': current_top,
                                    'page': page_num
                                })
                        current_line = ""
                        current_top = char['top']
                        line_size = char.get('size', 10)
                    current_line += char['text']
                    all_font_sizes.append(char.get('size', 10))

                if current_line.strip():
                    cleaned_line = clean_text_with_regex(current_line, noise_patterns)
                    if cleaned_line.strip():
                        page_items.append({
                            'type': 'text',
                            'content': cleaned_line,
                            'size': line_size,
                            'top': current_top,
                            'page': page_num
                        })

    # --- 4. 동적 헤더 임계값 계산 ---
    if not all_font_sizes:
        header_font_threshold = 18
    else:
        header_font_threshold = np.percentile(all_font_sizes, header_percentile)
        # 헤더와 본문이 동일한 경우 대비
        if header_font_threshold == max(all_font_sizes):
            header_font_threshold = max(all_font_sizes) - 1

    # --- 5. 1차 청킹 (헤더 기준) ---
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
                    font_size_chunks.append({
                        "header": current_chunk_header,
                        "content": current_chunk_content.strip()
                    })
                current_chunk_header = text
                current_chunk_content = ""
            else:
                current_chunk_content += text + "\n"

        elif item['type'] == 'table':
            current_chunk_content += "\n" + item['content'] + "\n"

    if current_chunk_content.strip():
        font_size_chunks.append({
            "header": current_chunk_header,
            "content": current_chunk_content.strip()
        })

    # --- 6. 2차 청킹 (사이즈 기준) ---
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=final_chunk_size,
        chunk_overlap=final_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    final_documents = []
    for chapter in font_size_chunks:
        content = chapter['content']
        sub_chunks = recursive_splitter.split_text(content)
        for sub_chunk_content in sub_chunks:
            def is_valid_header(line):
                blacklist = {'□', '※', '•', '-', '*', '·'}
                line_stripped = line.strip()
                if len(line_stripped) < 2:
                    return False
                if line_stripped in blacklist:
                    return False
                special_chars = set('`~!@#$%^&*()_+-=[]{}|;:\",./<>?·')
                total = len(line_stripped)
                if total == 0:
                    return False
                special_count = sum(1 for c in line_stripped if c in special_chars)
                if (special_count / total) > 0.6:
                    return False
                return True

            lines = [line.strip() for line in sub_chunk_content.split('\n') if line.strip()]
            valid_lines = [line for line in lines if is_valid_header(line)]
            if valid_lines:
                chunk_header = valid_lines[0]
            elif lines:
                chunk_header = lines[0]
            else:
                chunk_header = chapter['header']
            # --- 테이블 청크 또는 테이블 포함 청크는 예외 처리 ---
            if "table" not in chunk_header.lower() and "|" not in sub_chunk_content:
                if is_high_special_char_ratio(sub_chunk_content):
                    continue
            final_metadata = metadata.copy()
            final_metadata['parent_header'] = chunk_header
            doc = Document(page_content=sub_chunk_content, metadata=final_metadata)
            final_documents.append(doc)

    return final_documents

def is_high_special_char_ratio(text: str, threshold: float = 0.6) -> bool:
    SPECIAL_CHARS = set(string.punctuation + '`~!@#$%^&*()_+-=[]{}|;:",./<>?·')
    # 예외 문자 제외
    EXCEPT_CHARS = set('□※•')
    text_for_check = ''.join(c for c in text if c not in EXCEPT_CHARS)
    total_length = len(text_for_check)
    if total_length == 0:
        return False
    special_char_count = sum(1 for c in text_for_check if c in SPECIAL_CHARS)
    return (special_char_count / total_length) > threshold