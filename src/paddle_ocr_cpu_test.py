import os
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

# 1. OCR 엔진 초기화 (한국어+영어 모델 예시)
ocr = PaddleOCR(use_angle_cls=True, lang='korean')  

# 2. PDF -> 이미지 변환
pdf_path = "data/pdf/고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf"
pages = convert_from_path(pdf_path, dpi=300)
print(len(pages))

# 3. 각 페이지 OCR 실행
for page_num, page in enumerate(pages, start=1):
    # 임시 이미지 저장
    img_path = f"page_{page_num}.png"
    page.save(img_path, "PNG")

    # OCR 실행
    result = ocr.ocr(img_path, cls=True)

    print(f"\n=== Page {page_num} ===")
    for line in result[0]:
        text = line[1][0]   # 인식된 텍스트
        score = line[1][1]  # 신뢰도
        print(f"{text} (score: {score:.2f})")

    # 필요 없으면 임시 이미지 삭제
    os.remove(img_path)
