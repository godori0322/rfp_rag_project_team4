from document import chunk
from config import Config  
import os               

# 테스트 코드
# 해당 파일에서 HWP 원본 파일에서는 텍스트로 인식되는 부분이 PDF 변환 시 이미지로 변환되는 문제 있음. 

# 제목 텍스트 추출이 안되는 파일

## (사)벤처기업협회_2024년 벤처확인종합관리시스템 기능 고도화 용역사업 .pdf

# 검색이 안되는 파일

## 케빈랩 주식회사_평택시 강소형 스마트시티 AI 기반의 영상감시 시스템 .pdf (질문: 케빈랩 주식회사가 진행하는 사업에 대해 알려줘.)

filename = "케빈랩 주식회사_평택시 강소형 스마트시티 AI 기반의 영상감시 시스템 .pdf"

# Config를 사용하여 PDF 파일의 전체 경로 생성
# load_documents 함수와 동일한 방식
file_full_path = os.path.join(Config.PDF_PATH, filename)


if __name__ == "__main__":
    # chunk 함수에 파일 이름(filename) 대신 전체 경로(file_full_path)를 전달
    
    result = chunk(file_full_path, metadata={})
    cnt = 1
    
    # 테스트를 위해 프린트문 같이 출력
    print(f"'{filename}' 파일에 대한 청킹 결과:")
    print(f"🎉 총 {len(result)}개의 청크가 생성되었습니다.\n")
    for r in result:
        print(f"---- {cnt}번째 chunk ----")
        
        # 테스트를 위해 메타데이터에 추가된 parent_header도 함께 출력
        print(f"Parent Header: {r.metadata.get('parent_header', 'N/A')}") 
        print(r.page_content)
        cnt += 1
        if cnt > 30:
            break