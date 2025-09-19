from document import chunk

# 테스트 코드
# 해당 파일에서 HWP 원본 파일에서는 텍스트로 인식되는 부분이 PDF 변환 시 이미지로 변환되는 문제 있음. 

# 제목 텍스트 추출이 안되는 파일

## (사)벤처기업협회_2024년 벤처확인종합관리시스템 기능 고도화 용역사업 .pdf

# 검색이 안되는 파일

## 케빈랩 주식회사_평택시 강소형 스마트시티 AI 기반의 영상감시 시스템 .pdf (질문: 케빈랩 주식회사가 진행하는 사업에 대해 알려줘.)

filename = "케빈랩 주식회사_평택시 강소형 스마트시티 AI 기반의 영상감시 시스템 .pdf"

if __name__ == "__main__":
    result = chunk(filename, metadata={})
    cnt = 1
    for r in result:
        print(f"---- {cnt}번째 chunk ----")
        print(r.page_content)
        cnt += 1
        if cnt > 30:
            break