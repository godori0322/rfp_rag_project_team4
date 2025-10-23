# PaddleOCR-VL은 ERNIE-4.5-0.3B 언어 모델과 통합된 0.9B VLM을 핵심으로 하며, 109개 언어를 지원하고 텍스트, 표, 수식, 차트 등 복잡한 요소를 인식하는 데 SOTA 성능을 달성합니다. 이 파이프라인 역시 구조화된 Markdown 출력을 지원합니다.

from paddleocr import PaddleOCRVL

# --- 1. 파이프라인 초기화 ---
pipeline = PaddleOCRVL(lang="korean")

# --- 2. 입력 PDF 파일 경로 설정 ---
pdf_file_path = "data/pdf/고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf"
output_vl_save_path = "data/output"

# --- 3. 문서 파싱 실행 ---
# predict() 메서드를 사용하여 파싱을 실행합니다.
output = pipeline.predict(pdf_file_path) # [3]

# --- 4. 결과 저장 (Markdown 및 JSON) ---
for res in output:
    res.print()
    # Markdown 형식으로 저장
    res.save_to_markdown(save_path=output_vl_save_path)
    # JSON 형식으로 저장
    res.save_to_json(save_path=output_vl_save_path)

print(f"PaddleOCR-VL 결과가 '{output_vl_save_path}'에 저장되었습니다.")