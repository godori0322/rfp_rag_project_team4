# chain_router.py
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, chain, RunnableBranch
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from typing import List, Dict
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema.output_parser import StrOutputParser
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever


class ChainRouter:
    def __init__(self, llm, retriever, vectorstore, tracer):
        self.llm = llm
        self.retriever = retriever
        self.vectorstore = vectorstore
        self.tracer = tracer

    def find_documents(self, question):
        return self.retriever.get_relevant_documents(question)
    
    def find_contexts(self, docs):
        return [(doc.page_content + '\n' + json.dumps(doc.metadata, ensure_ascii=False)) for doc in docs]

    def get_recent_history(self, history, window_size=5):
        return history[-window_size:] if history else []

    def create_router_chain(self):
        # 컨텍스트 프롬프트: 이전 대화 맥락 유지
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 대화 맥락을 반영해서 사용자의 현재 질문을 독립적인 형태로 재구성하는 도우미야."),
            MessagesPlaceholder("history"),
            ("human", "{input}")
        ])

        contextualizer_chain = contextualize_prompt | self.llm | StrOutputParser()

        # 라우터 프롬프트: 쿼리 의도 분류
        route_prompt = ChatPromptTemplate.from_template(
            """사용자의 질문을 분석하여, 질문의 의도를 다음 5가지 카테고리 중 하나로 분류하세요.
            `general`: RFP 문서와 관련 없는 일반적인 질문 (인사, 농담, 상식 등).
            `summarization`: 문서의 핵심 내용을 요약해달라는 요청.
            `comparison`: 두 개 이상의 문서 내용을 비교하거나, 단일 문서내의 내용을 비교해달라는 요청.
            `recommendation`: 특정 사업과 유사한 다른 사업을 추천해달라는 요청.
            `default_qa`: RFP 문서 내용에 대한 구체적인 질문 또는 위 카테고리에 속하지 않는 모든 RFP 관련 질문.

            질문: {input}
            분류:"""
        )
        
        route_chain = route_prompt | self.llm | StrOutputParser()
        
    # 3단계: 각 의도에 맞는 전문화된 체인 정의
        general_chain = self._create_general_chain()
        summarization_chain = self._create_summarization_chain()
        comparison_chain = self._create_comparison_chain()
        recommendation_chain = self._create_recommendation_chain()
        default_qa_chain = self._create_default_qa_chain()

        # 4단계: 전체 파이프라인 결합 - 안정적인 데이터 흐름 보장
        def log_and_pass_through(data):
            print(f"✅ 라우터 분류 결과: {data['classification'].strip()}")
            return data # 입력받은 데이터를 그대로 반환

        full_chain = (
            RunnablePassthrough.assign(
                history=lambda x: self.get_recent_history(x.get("history", []))
            )
            | RunnablePassthrough.assign(refined_query=contextualizer_chain)
            .assign(classification=lambda x: route_chain.invoke({"input": x['refined_query']}))
            | RunnableLambda(log_and_pass_through) # [수정] 안전한 로깅 함수 사용
            | RunnableBranch(
                (lambda x: "general" in x["classification"], general_chain),
                (lambda x: "summarization" in x["classification"], summarization_chain),
                (lambda x: "comparison" in x["classification"], comparison_chain),
                (lambda x: "recommendation" in x["classification"], recommendation_chain),
                default_qa_chain
            )
        )
        
        return full_chain.with_config({"callbacks": [self.tracer]})
        

    def _create_general_chain(self):
        """## 일반 대화 체인 (Route 1)"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 RFP 문서에 대한 내용이 아니거나, 이해할 수 없는 문장이나 질문이면 '다시 질문해주세요.'라고 솔직하게 말해야해. "),
            ("human", "{refined_query}")
        ])
        return prompt | self.llm | StrOutputParser()

    def _create_default_qa_chain(self):
        """## RFP 관련 기본 QA 체인 (Route 2, Fallback Route) - 메타데이터 참조 강화"""
        
        def get_context_with_metadata(question: str) -> str:
            """질문과 관련된 문서를 검색하고, 각 문서의 메타데이터와 본문을 결합하여 컨텍스트를 생성합니다."""
            # 1. 질문을 기반으로 관련 문서를 검색합니다.
            docs = self.find_documents(question)
            if not docs:
                print("--- WARNING (Default QA): 관련된 문서를 찾지 못했습니다. ---")
                return "관련된 문서를 찾을 수 없습니다."
            
            retrieved_titles = [doc.metadata.get('project_title', '제목 없음') for doc in docs]
            print(f"--- INFO (Default QA): 검색된 문서 목록: {list(set(retrieved_titles))} ---")

            # 2. 각 문서의 내용과 메타데이터를 결합하여 컨텍스트 문자열 리스트를 생성합니다.
            context_parts = []
            for doc in docs:
                # 메타데이터를 JSON 형식으로 보기 좋게 변환합니다.
                metadata_str = json.dumps(doc.metadata, ensure_ascii=False, indent=2)
                # LLM이 문서의 정보와 본문을 명확히 구분할 수 있도록 형식을 지정합니다.
                part = (
                    f"--- 문서 시작 ---\n"
                    f"[문서 정보]:\n{metadata_str}\n\n"
                    f"[문서 본문]:\n{doc.page_content}\n"
                    f"--- 문서 종료 ---"
                )
                context_parts.append(part)
            
            # 3. 모든 컨텍스트 부분을 하나의 문자열로 결합하여 반환합니다.
            return "\n\n".join(context_parts)

        # LLM에게 역할을 부여하고, 메타데이터 활용법을 명시적으로 지시하는 프롬프트
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "당신은 대한민국 B2G(정부 대상 사업) RFP(제안요청서) 분석을 전문으로 하는 아주 똑똑하고 정확한 AI 컨설턴트입니다.\n"
                "당신의 임무는 주어진 [컨텍스트]만을 근거로 사용자의 [질문]에 대해 체계적이고 사실에 입각하여 답변하는 것입니다.\n\n"
                "**작업 절차:**\n"
                "1. 사용자의 [질문]의 핵심 의도를 정확히 파악합니다.\n"
                "2. 제공된 [컨텍스트] 내의 여러 문서들을 빠르게 스캔하여, 질문과 가장 관련 있는 내용이 담긴 [문서 본문]을 찾습니다.\n"
                "3. 찾은 내용을 바탕으로 답변을 생성합니다.\n\n"
                "**답변 생성 규칙:**\n"
                "- **출처 명시:** 답변의 신뢰도를 높이기 위해, 근거로 사용한 문서의 [문서 정보]에 있는 'project_title' 또는 'rfp_number'를 반드시 언급해야 합니다. (예: 'OO 사업(공고번호: 123)에 따르면...')\n"
                "- **사실 기반:** 오직 [컨텍스트]에 명시된 내용만을 근거로 답변해야 하며, 절대 당신의 사전 지식을 사용하거나 정보를 추측해서는 안 됩니다.\n"
                "- **정보 부재 시:** 만약 [컨텍스트]에서 질문에 대한 답을 명확하게 찾을 수 없다면, '제공된 문서에서는 해당 정보를 확인할 수 없습니다.'라고만 답변해야 합니다."
            ),
            ("human", "[질문]: {question}\n\n[컨텍스트]:\n{context}")
        ])

        # 체인 구성: 입력(질문) -> 컨텍스트 생성 -> 프롬프트 조합 -> LLM 답변 -> 문자열 출력
        return (
            {
                "context": lambda x: get_context_with_metadata(x["refined_query"]),
                "question": lambda x: x["refined_query"]
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _create_summarization_chain(self):
        """## 변경된 로직: 정보 요약 체인 (Route 3) - LCEL 기반 Map-Reduce"""
        
        # 1. 검색 및 문서 분할 (get_documents_with_metadata는 기존 로직)
        def get_documents_with_metadata(x):
            docs = self.find_documents(x['refined_query'])
            # 메타데이터를 텍스트에 포함시켜 반환
            docs = [Document(page_content=f"메타데이터: {json.dumps(doc.metadata, ensure_ascii=False, indent=2)}\n\n문서 내용: {doc.page_content}") for doc in docs]
            for doc in docs:
                print(f'{doc.page_content}\n')
            return docs

        # 2. Map 단계 프롬프트
        map_prompt_template = """
        다음은 RFP 문서의 일부 내용입니다. 이 내용에서 핵심 정보를 요약하세요.
        ---
        {text}
        ---
        요약:
        """
        map_prompt = ChatPromptTemplate.from_template(map_prompt_template)
        map_chain = map_prompt | self.llm | StrOutputParser()

        # 3. Reduce 단계 프롬프트
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
        
        # 4. 전체 파이프라인 결합
        # 문서 검색 -> 각 문서를 요약(map) -> 요약들을 합쳐 최종 요약(reduce)
        summarize_pipeline = (
            RunnableLambda(get_documents_with_metadata)
            | map_chain.map()  # 모든 문서에 대해 map_chain 실행
            | reduce_chain      # map 결과를 reduce_chain에 전달
        )
        
        return summarize_pipeline

    
    def _create_comparison_chain(self):
        """## 비교 분석 체인 (단일/다중 문서 처리)"""
        # 1단계: 비교 유형 분류 및 정보 추출(Triage) 체인
        triage_parser = JsonOutputParser()
        triage_prompt = ChatPromptTemplate.from_template(
            """ 당신은 두 개 또는 한 개의 사업(RFP) 정보를 받아서, 주어진 기준에 따라 명확하게 비교 분석하는 전문 컨설턴트입니다.
                사용자의 비교 질문을 분석하여 'multi_document' 또는 'single_document' 유형으로 분류하고 관련 정보를 추출하세요.
                - 'multi_document': 서로 다른 두 문서를 비교. 'item_A', 'item_B'를 추출.
                - 'single_document': 하나의 문서 내에서 두 개념을 비교. 기준이 되는 'base_document'와 두 개념 'topic_A', 'topic_B'를 추출.
                - 'criteria': 비교 기준을 추출. 명확하지 않으면 "전반적인 특징"으로 설정.
                오직 JSON 객체로만 답변하세요.

                예시 1 (multi_document):
                질문: "A사업과 B사업을 비교해줘"
                JSON: {{"type": "multi_document", "item_A": "A사업", "item_B": "B사업", "criteria": "전반적인 특징"}}

                예시 2 (single_document):
                질문: "부산관광공사 사업에서 '대결 기능'과 '협조 기능'을 비교해줘"
                JSON: {{"type": "single_document", "base_document": "부산관광공사 사업", "topic_A": "대결 기능", "topic_B": "협조 기능", "criteria": "전반적인 특징"}}

                사용자 질문: {question}
                JSON 출력: """
        )
        # 이제 'refined_query'를 입력으로 받습니다.
        triage_chain = {"question": lambda x: x['refined_query']} | triage_prompt | self.llm | triage_parser

        # 2-1: 다중 문서 비교 로직
        def get_context_for_multi_doc(item_name: str) -> str:
            print(f"--- INFO (Multi-Doc): '{item_name}'에 대한 지능형 문서 검색 수행 ---")
            # self.retriever (SelfQueryRetriever)는 이름이 조금 달라도 유연하게 찾아낼 수 있습니다.
            docs = self.retriever.invoke(item_name)
            if not docs: return f"'{item_name}' 정보를 찾을 수 없습니다."
            return "\n\n".join(self.find_contexts(docs))

        multi_doc_prompt = ChatPromptTemplate.from_template(
            "**비교 기준:** {criteria}\n\n"
            "**[사업 A: {item_A}]**\n{context_A}\n\n"
            "**[사업 B: {item_B}]**\n{context_B}\n\n"
            "위 두 사업의 정보를 바탕으로, '비교 기준'에 맞춰 각각의 내용을 요약하고 비교 결과를 표(Markdown Table) 형식으로 명확하게 제시해주십시오. 만약 특정 정보를 찾을 수 없다면 '정보 확인 불가' 라고 명시하세요."
        )
        
        multi_doc_chain = RunnablePassthrough.assign(
            item_A=lambda x: x["triage_result"]["item_A"], item_B=lambda x: x["triage_result"]["item_B"],
            criteria=lambda x: x["triage_result"]["criteria"],
            context_A=lambda x: get_context_for_multi_doc(x["triage_result"]["item_A"]),
            context_B=lambda x: get_context_for_multi_doc(x["triage_result"]["item_B"])
        ) | multi_doc_prompt | self.llm | StrOutputParser()
        
        # 2-2: 단일 문서 비교 로직
        def retrieve_and_extract_for_single_doc(input_dict: dict) -> dict:
            # 큰 서류철(input_dict)에서 '비교 정보' 폴더(triage_result)를 먼저 꺼냅니다.
            triage_result = input_dict["triage_result"]
            
            # 이제 '비교 정보' 폴더 안에서 필요한 서류를 찾습니다.
            base_doc_name = triage_result["base_document"]
            topic_A = triage_result["topic_A"]
            topic_B = triage_result["topic_B"]
            
            print(f"--- INFO (Single-Doc): 기준 문서 '{base_doc_name}'의 컨텍스트 검색 ---")
            base_docs = self.retriever.invoke(base_doc_name)
            if not base_docs: return {"extracted_snippets": f"기준 문서 '{base_doc_name}'를 찾을 수 없습니다.", **triage_result}
            
            local_context = "\n\n".join(self.find_contexts(base_docs))
            
            print(f"--- INFO (Single-Doc): '{topic_A}'와 '{topic_B}'에 대한 정보 추출 ---")
            extraction_prompt = ChatPromptTemplate.from_template("문서 내용에서 '{topic_A}'와 '{topic_B}'에 대한 부분을 각각 찾아서 요약해줘.\n---\n{context}\n---")
            snippet_extractor = extraction_prompt | self.llm | StrOutputParser()
            extracted_snippets = snippet_extractor.invoke({"context": local_context, "topic_A": topic_A, "topic_B": topic_B})
            
            return {"extracted_snippets": extracted_snippets, **triage_result}

        single_doc_prompt = ChatPromptTemplate.from_template(
            "**기준 문서:** {base_document}\n**비교 기준:** {criteria}\n\n**추출된 정보:**\n---\n{extracted_snippets}\n---\n\n위 정보를 바탕으로 '{topic_A}'와 '{topic_B}'를 비교 분석하고 요약해줘."
        )
        single_doc_chain = RunnableLambda(retrieve_and_extract_for_single_doc) | single_doc_prompt | self.llm | StrOutputParser()

        branch = RunnableBranch(
            (lambda x: x["triage_result"].get("type") == "single_document", single_doc_chain),
            (lambda x: x["triage_result"].get("type") == "multi_document", multi_doc_chain),
            (lambda x: "비교 유형을 식별할 수 없습니다.")
        )
        return RunnablePassthrough.assign(triage_result=triage_chain) | branch






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