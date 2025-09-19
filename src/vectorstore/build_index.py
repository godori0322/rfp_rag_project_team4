from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List

from config import OPENAI_API_KEY, EMBEDDING_MODEL, VECTOR_DB_PATH

def build_and_save_index(docs: List[str], save_path: str = VECTOR_DB_PATH, embedding_model: str = EMBEDDING_MODEL):
    if OPENAI_API_KEY is None:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일 확인 필요")

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_key=OPENAI_API_KEY
    )

    vector_store = FAISS.from_texts(docs, embeddings)
    vector_store.save_local(save_path)

    print(f"Vector Store 저장 완료: {save_path}")

if __name__ == "__main__":
    sample_docs = [
        "이러닝 시스템은 강의 콘텐츠 업로드, 사용자 진도 관리, 퀴즈 및 평가 기능, 학습 통계 제공, 사용자 맞춤형 추천 기능 등을 갖추어야 합니다. 또한 모바일과 웹 환경에서 모두 접근 가능해야 하며, 보안과 개인정보 보호도 필수 요건입니다.",
        "사무실 내 네트워크는 안정적인 와이파이 환경과 유선 연결을 제공해야 하며, 방화벽과 VPN 설정을 통해 외부 공격으로부터 보호해야 합니다. 프린터와 공유 서버 관리도 포함됩니다.",
        "연례 행사는 팀 빌딩과 직원 사기 증진을 목표로 하며, 장소 예약, 행사 일정, 예산 계획, 식사 및 기념품 준비가 필요합니다. 참여 인원 관리와 피드백 수집도 필수입니다."
    ]
    build_and_save_index(sample_docs)