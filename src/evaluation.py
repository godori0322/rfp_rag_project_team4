import os
import json
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    AnswerSimilarity,
    ContextRecall,
    ContextPrecision,
    AnswerCorrectness,
)
from langchain_openai import ChatOpenAI
from ragas.embeddings import OpenAIEmbeddings
from openai import AsyncOpenAI # 수정: AsyncOpenAI 임포트
from langchain_core.messages import HumanMessage, AIMessage

from config import Config
from chatbot import Chatbot

# 환경 변수 로드
load_dotenv(find_dotenv())

# RAGAS 평가를 위한 LLM 및 Embeddings 설정
ragas_llm_base = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
ragas_llm = LangchainLLMWrapper(langchain_llm=ragas_llm_base)

# 수정: 비동기 openai 클라이언트를 인스턴스화
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 수정: ragas.embeddings.OpenAIEmbeddings에 비동기 client 인자를 전달
ragas_embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-large")

# RAGAS 메트릭 초기화
metrics = [
    Faithfulness(llm=ragas_llm),
    AnswerRelevancy(llm=ragas_llm),
    AnswerSimilarity(),
    ContextRecall(llm=ragas_llm),
    ContextPrecision(llm=ragas_llm),
    AnswerCorrectness(llm=ragas_llm),
]

def generate_ragas_dataset(test_questions_with_ground_truths: list[dict]):
    """
    제공된 테스트 질문과 정답을 기반으로 RAGAS 평가용 데이터셋을 생성합니다.
    """
    ragas_data = { 'question': [], 'answer': [], 'contexts': [], 'reference': []}   

    for item in test_questions_with_ground_truths:
        question = item['question']
        ground_truth = item['ground_truth']
        bot = Chatbot(user_id="ragas_eval_user")

        print('======= chatbot_input ==========')
        print(question)
        
        try:
            bot_response = bot.ask(question, False)
            bot_contexts = bot.find_contexts(bot.find_documents(question))
        except Exception as e:
            print(f"첫 번째 시도 실패: {e}")
            try:
                print("두 번째 시도 (재호출)...")
                bot_response = bot.ask(question, False)
                bot_contexts = bot.find_contexts(bot.find_documents(question))
                print("두 번째 시도 성공.")
            except Exception as e:
                print(f"두 번째 시도 실패: {e}")
                print("재호출 실패. 처리를 중단합니다.")

        print('======= chatbot_response ==========')
        print(bot_response)
        print('===================================')
        
        ragas_data['question'].append(question)
        ragas_data['answer'].append(bot_response)
        ragas_data['contexts'].append(bot_contexts)
        ragas_data['reference'].append(ground_truth)

    return Dataset.from_dict(ragas_data)

if __name__ == "__main__":
    # 평가용 질문과 정답 쌍 (예시)
    # test_questions_with_ground_truths = [
    #     {'question': "고려대학교 차세대 포털·학사 정보시스템 구축 사업의 사업 기간과 무상유지보수 기간은 각각 어떻게 되나요?", 
    #      'ground_truth': "사업 기간은 계약일로부터 24개월 이내이며, 무상유지보수 기간은 사업 종료일로부터 12개월입니다."},
    #     {'question': "고려대학교 차세대 포털·학사 정보시스템 구축 사업의 주요 범위 중 '포털시스템'에 포함되는 주요 내용에는 어떤 것들이 있나요?", 
    #      'ground_truth': "포털시스템의 주요 내용은 통합로그인, 통합/지능형 검색, 마이페이지, 공지/알림, 일정관리, 커뮤니티, 게시판, 사용자별 정보서비스, 위젯, 연계서비스(웹메일, 챗봇, 전자결재, 학사/행정 서비스) 등입니다. 또한 학생(졸업생 포함), 교직원, 연구원 등 내부 구성원을 대상으로 하며, 학생/교수 등 신분별 개인별 주요 정보 제공, 학사/행정/연구 시스템의 주요 기능 직접 접근 등을 포함합니다."},
    #     {'question': "제안서 작성 시 유의해야 할 주요 사항은 무엇인가요?", 
    #      'ground_truth': "제안서 작성 시 유의사항으로는 제안서의 효력 유지, 제안서 내용의 명확성, 제안서 구성의 일관성, 객관적인 증빙자료 제시, 제안서 내용 변경 금지, 기한 내 제출, 제출된 제안서의 권리 귀속 등이 있습니다. 또한, 제출된 제안서는 반환되지 않으며, 제안서 작성 및 제출과 관련된 비용은 제안사가 부담해야 합니다."},
    # ]
    df = pd.read_csv("data/evaluation.csv")
    test_questions_with_ground_truths = df.to_dict(orient='records')
   
    print("RAGAS 데이터셋 생성 중...")
    ragas_dataset = generate_ragas_dataset(test_questions_with_ground_truths)
    print("RAGAS 데이터셋 생성 완료.")

    print("RAGAS 평가 실행 중...")
    result = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    print("RAGAS 평가 완료.")
    print(result)
