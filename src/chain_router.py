# chain_router.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, chain
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from typing import List, Dict
#from langchain.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

class ChainRouter:
    def __init__(self, llm, retriever, vectorstore, tracer, find_documents_func, find_contexts_func):
        self.llm = llm
        self.retriever = retriever
        self.vectorstore = vectorstore
        self.tracer = tracer
        self.find_documents = find_documents_func
        self.find_contexts = find_contexts_func

    def create_router_chain(self):
        """## 변경된 로직: 라우팅 로직을 @chain을 사용한 함수로 처리합니다."""
        # 라우터 프롬프트: 쿼리 의도 분류
        route_prompt = ChatPromptTemplate.from_template(
            """사용자의 질문을 분석하여, 질문의 의도를 다음 5가지 카테고리 중 하나로 분류하세요.
            `metadata_search`: 특정 조건(기관, 사업명 등)으로 문서를 찾아달라는 요청.
            `summarization`: 문서의 핵심 내용을 요약해달라는 요청.
            `specific_qa`: 단일 문서에 대한 구체적인 정보(사실, 수치)를 묻는 질문.
            `comparison`: 두 개 이상의 문서나 항목을 비교해달라는 요청.
            `recommendation`: 특정 사업과 유사한 다른 사업을 추천해달라는 요청.
            
            질문: {input}
            분류:"""
        )
        
        route_chain = route_prompt | self.llm | StrOutputParser()

        @chain
        def router_chain_optimized(input_dict):
            """질문을 한번만 분류하고, 그 결과를 바탕으로 분기를 선택합니다."""
            classification = route_chain.invoke({"input": input_dict["input"]})
            
            print(f"라우터 분류 결과: {classification.strip()}")
            
            if "metadata_search" in classification:
                return self._create_metadata_search_chain()
            elif "summarization" in classification:
                return self._create_summarization_chain()
            elif "specific_qa" in classification:
                return self._create_specific_qa_chain()
            elif "comparison" in classification:
                return self._create_comparison_chain()
            elif "recommendation" in classification:
                return self._create_recommendation_chain()
            else:
                return self._create_specific_qa_chain()

        return router_chain_optimized.with_config(
            {"callbacks": [self.tracer]}
        )

    def _create_metadata_search_chain(self):
        """## 메타데이터 검색 체인 (Route 1)"""
        return (
            RunnablePassthrough.assign(docs=RunnableLambda(self.find_documents))
            | RunnableLambda(
                lambda x: "다음은 검색된 RFP 문서 목록입니다.\n\n" + "\n\n".join([
                    f"- {doc.metadata.get('project_title', '제목 없음')} (공고번호: {doc.metadata.get('rfp_number', '미상')})"
                    for doc in x['docs']
                ])
            )
        )

    def _create_summarization_chain(self):
        """## 정보 요약 체인 (Route 2)"""
        # 긴 문서를 처리하기 위해 map-reduce 방식 등을 고려할 수 있음
        summarization_prompt = ChatPromptTemplate.from_messages([
            ("system", "제공된 RFP 문서의 내용을 바탕으로 핵심 정보를 간결하게 요약하세요."),
            ("human", "문서 내용:\n{context}")
        ])
        def find_contexts(x):
            contexts = self.find_contexts(self.find_documents(x['input']))
            for context in contexts:
                print(context)
                print()
            return contexts
        
        return (
            RunnablePassthrough.assign(context=RunnableLambda(find_contexts)) # lambda x: self.find_contexts(self.find_documents(x['input']))
            | summarization_prompt
            | self.llm
            | StrOutputParser()
        )

    def _create_comparison_chain(self):
        """## [개선] 비교 분석 체인 (Route 4) - 리트리버 일원화"""

        extraction_parser = JsonOutputParser()
        extraction_prompt = ChatPromptTemplate.from_template(
            """당신은 사용자의 복잡한 질문을 분석하여 구조화된 JSON으로 변환하는 전문 분석가입니다.
            사용자의 질문에서 비교할 두 개의 대상(item_A, item_B)과 비교 기준(criteria)을 추출해야 합니다.
            만약 기준이 명확하지 않다면 "전반적인 특징"이라고 답하십시오.
            답변은 반드시 JSON 형식으로만 제공해야 합니다. 다른 설명은 절대 추가하지 마십시오.

            사용자 질문: {input}
            JSON 출력:
            """
        )
        extraction_chain = extraction_prompt | self.llm | extraction_parser

        # get_context_for_item_robust : 일원화된 메인 Retriever를 사용하여 각 아이템을 검색하는 함수
        def get_context_for_item_robust(item_name: str) -> str:
            """[개선] ChainRouter에 전달된 기본 self.retriever를 사용하여 검색하고, 실패 시 DB를 직접 확인합니다."""
            print(f"--- INFO: '{item_name}'에 대한 메인 SelfQueryRetriever 검색 수행 ---")
            
            # Chatbot 클래스에서 생성되어 __init__을 통해 전달받은 self.retriever를 직접 사용
            docs = self.retriever.invoke(item_name)
            
            if not docs:
                print(f"--- WARNING: '{item_name}'에 대한 검색 실패 ---")
                
                # ### DEBUGGING BLOCK START ###
                print("--- DEBUGGING: DB 데이터 샘플 직접 확인 ---")
                try:
                    # 벡터 DB에서 직접 메타데이터 샘플을 가져온다
                    sample_data = self.vectorstore.get(limit=5, include=["metadatas"])
                    sample_titles = [
                        meta.get('project_title', '제목 없음') 
                        for meta in sample_data.get('metadatas', [])
                    ]
                    print("DB에 저장된 프로젝트 제목 샘플:", sample_titles)
                except Exception as e:
                    print(f"DB 샘플 데이터 조회 중 오류 발생: {e}")
                # ### DEBUGGING BLOCK END ###

                return "관련 정보를 찾을 수 없습니다. (디버깅 정보가 콘솔에 출력되었습니다)"
                
            return "\n\n".join(self.find_contexts(docs))
        
        
        comparison_prompt = ChatPromptTemplate.from_template(
            """당신은 두 개의 사업(RFP) 정보를 받아서, 주어진 기준에 따라 명확하게 비교 분석하는 전문 컨설턴트입니다.

            **비교 기준:** {criteria}

            **[사업 A: {item_A}]**
            {context_A}

            **[사업 B: {item_B}]**
            {context_B}

            **지시사항:**
            위 두 사업의 정보를 바탕으로, '비교 기준'에 맞춰 각각의 내용을 요약하고 비교 결과를 표(Markdown Table) 형식으로 명확하게 제시해주십시오.
            만약 특정 정보를 찾을 수 없다면 "정보 확인 불가"라고 명시하세요.
            """
        )
        
        # 전체 체인 결합
        return (
            RunnablePassthrough.assign(extracted_info=extraction_chain)
            | RunnablePassthrough.assign(
                item_A = lambda x: x["extracted_info"]["item_A"],
                item_B = lambda x: x["extracted_info"]["item_B"],
                criteria = lambda x: x["extracted_info"]["criteria"],
                context_A=lambda x: get_context_for_item_robust(x["extracted_info"]["item_A"]),
                context_B=lambda x: get_context_for_item_robust(x["extracted_info"]["item_B"])
            )
            | comparison_prompt
            | self.llm
            | StrOutputParser()
        )

    def _create_recommendation_chain(self):
        """## 유사 사업 추천 체인 (Route 5) - [Query Expansion 적용]"""

        # Query Expansion 체인 정의
        query_expansion_prompt = ChatPromptTemplate.from_template(
            """당신은 사용자의 질문을 벡터 검색에 더 효과적인 키워드 목록으로 확장하는 전문가입니다.
            사용자의 원본 질문의 핵심 의미를 파악하여, 관련 동의어, 기술 용어, 상위 개념 등을 포함한 검색 키워드 3개를 쉼표(,)로 구분하여 생성하세요.

            사용자 질문: {input}
            검색 키워드:"""
        )

        query_expansion_chain = query_expansion_prompt | self.llm | StrOutputParser()

        # 헬퍼 함수
        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(
                f"### 사업명: {doc.metadata.get('project_title', '제목 없음')}\n"
                f"공고번호: {doc.metadata.get('rfp_number', '미상')}\n"
                f"요약:\n{doc.metadata.get('summary', doc.page_content)}"
                for doc in docs
            )
            
        recommendation_prompt = ChatPromptTemplate.from_template(
            """당신은 사용자의 요청에 기반하여 유사한 사업을 추천하고, 그 이유를 설명하는 B2G 입찰 전문 컨설턴트입니다.

            **사용자 원본 요청:**
            {original_input}

            **검색된 유사 사업 목록:**
            {context}

            **지시사항:**
            '검색된 유사 사업 목록'을 바탕으로, 각 사업이 왜 '사용자 원본 요청'과 유사한지 핵심 이유를 설명하며 추천 목록을 작성해주십시오.
            """
        )
        
        mmr_retriever = self.vectorstore.as_retriever(search_type="mmr")

        # 전체 체인 결합
        recommendation_chain = (
            {
                # expanded_query 키에 확장된 쿼리 결과를 할당
                "expanded_query": query_expansion_chain,
                # original_input 키에 원본 사용자 입력을 그대로 유지
                "original_input": (lambda x: x["input"])
            }
            | RunnablePassthrough.assign(
                # expanded_query로 문서를 검색하고 그 결과를 'context'에 할당함.
                context=lambda x: format_docs(mmr_retriever.get_relevant_documents(x["expanded_query"]))
            )
            | recommendation_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return recommendation_chain