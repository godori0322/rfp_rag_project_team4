from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL

class AnswerGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)
        self.prompt_template = """
        당신은 RFP(제안요청서) 분석 전문가입니다. 
        주어진 문서를 바탕으로 사용자의 질문에 대해 한국어로 답변해주세요.
        문서에 내용이 없으면 '문서에서 관련 정보를 찾을 수 없습니다.'라고 답변하세요.

        [문서]
        {context}

        [질문]
        {question}

        [답변]
        """
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)

    def generate(self, question: str, context: List[str]) -> str:
        """LLM을 사용하여 검색된 문서를 바탕으로 답변을 생성합니다."""
        chain = self.prompt | self.llm
        formatted_context = "\n---\n".join(context)
        response = chain.invoke({
            "question": question,
            "context": formatted_context
        })
        return response.content