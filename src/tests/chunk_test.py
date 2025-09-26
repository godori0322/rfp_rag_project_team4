#chunk_test.py
from document import chunk
from config import Config  
import os
import pandas as pd
import pdfplumber
import sys

# 절대경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 테스트할 파일
filename = "국민연금공단_2024년 이러닝시스템 운영 용역.hwp"


def ext(original_filename, ext='pdf'):
    base_filename, _ = os.path.splitext(original_filename)
    return f"{base_filename}.{ext}"
def get_total_pages(pdf_path: str) -> int:
    """
    ## 변경된 로직: pdfplumber를 사용해 PDF 파일의 총 페이지수를 반환합니다.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception as e:
        print(f"PDF 페이지 수 추출 중 오류 발생: {e}")
        return 0
def get_metadata_from_csv(csv_path: str, filename: str) -> dict:
    """
    CSV 파일에서 특정 파일명에 해당하는 메타데이터를 찾아 딕셔너리로 반환합니다.
    이때, '텍스트' 열은 제외합니다.
    """
    if not os.path.exists(csv_path):
        print(f"오류: {csv_path} 파일을 찾을 수 없습니다.")
        return {}
        
    try:
        df = pd.read_csv(csv_path)
        matching_row = df[df['파일명'] == filename]
        
        if not matching_row.empty:
            # '텍스트' 필드(컬럼)가 존재하면 제외
            if '텍스트' in matching_row.columns:
                matching_row = matching_row.drop('텍스트', axis=1)
                
            # 첫 번째 일치하는 행을 딕셔너리로 반환
            return matching_row.to_dict('records')[0]
        else:
            print(f"경고: CSV에서 '{filename}'에 해당하는 메타데이터를 찾을 수 없습니다.")
            return {}
    except Exception as e:
        print(f"메타데이터 로딩 중 오류 발생: {e}")
        return {}

if __name__ == "__main__":
    file_full_path = os.path.join(Config.PDF_PATH, ext(filename))
    csv_file_path = os.path.join("data", "raw", "data_list.csv")

    # chunk 함수에 파일 이름(filename) 대신 전체 경로(file_full_path)를 전달
    document_metadata = get_metadata_from_csv(csv_file_path, filename)
    # PDF 총 페이지수 가져오기
    total_pages = get_total_pages(file_full_path)

    if document_metadata:
        print("--- 청킹 시작 ---")
        print(f"파일: '{filename}'")        
        print(f"추가 메타데이터: {document_metadata}")

        result = chunk(file_full_path, metadata=document_metadata)
        print(f"총 페이지수: {total_pages} 페이지") # 총 페이지수 출력
        print(f"\n🎉 총 {len(result)}개의 청크가 생성되었습니다 (페이지당 청크: {len(result)/total_pages:.2f}).\n")
        
        cnt = 1
        for r in result:
            print(f"---- {cnt}번째 chunk ----")
            print(f"메타데이터 (parent_header): {r.metadata['parent_header']}\n###") 
            print(r.page_content)
            cnt += 1
            if cnt > 30: # 너무 많은 출력을 방지
                break
    else:
        print("메타데이터를 찾을 수 없으므로 청킹을 시작할 수 없습니다.")



# 테스트 코드
# 해당 파일에서 HWP 원본 파일에서는 텍스트로 인식되는 부분이 PDF 변환 시 이미지로 변환되는 문제 있음. 
# 제목 텍스트 추출이 안되는 파일

## (사)벤처기업협회_2024년 벤처확인종합관리시스템 기능 고도화 용역사업 .pdf

# 검색이 안되는 파일
## 케빈랩 주식회사_평택시 강소형 스마트시티 AI 기반의 영상감시 시스템 .pdf (질문: 케빈랩 주식회사가 진행하는 사업에 대해 알려줘.)
