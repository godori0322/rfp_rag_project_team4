# chain_router.py
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, chain
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from typing import List, Dict
from langchain.schema.output_parser import StrOutputParser
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document

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
        """## 변경된 로직: 정보 요약 체인 (Route 2) - LCEL 기반 Map-Reduce"""
        
        # 1. 대화 맥락을 고려하여 질문을 재구성하는 리트리버 체인
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "과거 대화 내용을 바탕으로 다음 질문을 재구성하세요. 단, 요약하려는 문서에 대한 정보를 구체적으로 포함해야 합니다."),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ])
        history_aware_retriever_chain = create_history_aware_retriever(self.llm, self.retriever, contextualize_prompt)

        # 2. 검색 및 문서 분할 (get_documents_with_metadata는 기존 로직)
        def get_documents_with_metadata(x):
            docs = self.find_documents(x['input'])
            # 메타데이터를 텍스트에 포함시켜 반환
            docs = [Document(page_content=f"메타데이터: {json.dumps(doc.metadata, ensure_ascii=False, indent=2)}\n\n문서 내용: {doc.page_content}") for doc in docs]
            for doc in docs:
                print(f'{doc.page_content}\n')
            return docs

        # 3. Map 단계 프롬프트
        map_prompt_template = """
        다음은 RFP 문서의 일부 내용입니다. 이 내용에서 핵심 정보를 요약하세요.
        ---
        {text}
        ---
        요약:
        """
        map_prompt = ChatPromptTemplate.from_template(map_prompt_template)
        map_chain = map_prompt | self.llm | StrOutputParser()

        # 4. Reduce 단계 프롬프트
        reduce_prompt_template = """
        다음은 여러 문서 청크에 대한 요약본입니다. 이 요약들을 종합하여 하나의 최종 요약을 생성하세요.
        최종 요약은 핵심 내용을 간결하고 명확하게 포함해야 합니다.
        ---
        {text}
        ---
        최종 요약:
        """
        reduce_prompt = ChatPromptTemplate.from_template(reduce_prompt_template)
        reduce_chain = reduce_prompt | self.llm | StrOutputParser()
        
        # 5. 전체 파이프라인 결합
        # 문서 검색 -> 각 문서를 요약(map) -> 요약들을 합쳐 최종 요약(reduce)
        summarize_pipeline = (
            # 대화 맥락을 고려한 검색
            RunnablePassthrough.assign(retrieved_docs=history_aware_retriever_chain)
            | RunnableLambda(get_documents_with_metadata)
            | map_chain.map()  # 모든 문서에 대해 map_chain 실행
            | reduce_chain      # map 결과를 reduce_chain에 전달
        )
        
        return summarize_pipeline

    def _create_specific_qa_chain(self):
        """## 구체적 질의응답 체인 (Route 3)"""
        # 변경된 로직: 프롬프트 변수명을 'input'과 'chat_history'로 변경
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "과거 대화 내용을 바탕으로 다음 질문을 재구성하세요."),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(self.llm, self.retriever, contextualize_prompt)
        
        qa_prompt = ChatPromptTemplate.from_template("""
        당신은 B2G 입찰지원 전문 컨설턴트 챗봇입니다.
        제공된 'Context'의 내용을 바탕으로 질문에 답변하세요.
        
        Context:
        {context}
        
        질문: {input}
        """)
        
        return create_retrieval_chain(history_aware_retriever, create_stuff_documents_chain(self.llm, qa_prompt))
    
    def _create_comparison_chain(self):
        """## 비교 분석 체인 (Route 4)"""
        # 여러 문서를 찾아 LLM에 전달하여 비교 요청
        comparison_prompt = ChatPromptTemplate.from_messages([
            ("system", "제공된 여러 RFP 문서의 내용을 바탕으로 사용자의 비교 요청에 응답하세요."),
            ("human", "문서 내용:\n{context}\n\n질문: {input}")
        ])
        return (
            RunnablePassthrough.assign(context=RunnableLambda(lambda x: self.find_contexts(self.find_documents(x['input']))))
            | comparison_prompt
            | self.llm
            | StrOutputParser()
        )

    def _create_recommendation_chain(self):
        """## 유사 사업 추천 체인 (Route 5) - [Query Expansion 적용]"""

        # 1. 질의 확장(Query Expansion)을 위한 체인 정의
        query_expansion_prompt = ChatPromptTemplate.from_template(
            """당신은 사용자의 질문을 벡터 검색에 더 효과적인 키워드 목록으로 확장하는 전문가입니다.
            사용자의 원본 질문의 핵심 의미를 파악하여, 관련 동의어, 기술 용어, 상위 개념 등을 포함한 검색 키워드 3개를 쉼표(,)로 구분하여 생성하세요.

            사용자 질문: {input}
            검색 키워드:"""
        )

        query_expansion_chain = query_expansion_prompt | self.llm | StrOutputParser()

        # 2. 기존 추천 체인 로직 (헬퍼 함수 등)
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

        # 3. 전체 체인 결합
        recommendation_chain = (
            {
                # 'expanded_query' 키에 확장된 쿼리 결과를 할당
                "expanded_query": query_expansion_chain,
                # 'original_input' 키에 원본 사용자 입력을 그대로 유지
                "original_input": (lambda x: x["input"])
            }
            | RunnablePassthrough.assign(
                # 확장된 쿼리('expanded_query')를 사용해 문서를 검색하고, 그 결과를 'context'에 할당
                context=lambda x: format_docs(mmr_retriever.get_relevant_documents(x["expanded_query"]))
            )
            | recommendation_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return recommendation_chain