# chain_router.py
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from typing import List, Dict, Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema.output_parser import StrOutputParser
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever


class ChainRouter:
    def __init__(self, llm, retriever, self_query_retriever, vectorstore, tracer):
        self.llm = llm
        self.retriever = retriever
        self.self_query_retriever = self_query_retriever
        self.vectorstore = vectorstore
        self.tracer = tracer

    def find_documents(self, question):
        print(f"--- INFO: '{question}'에 대한 지능형 문서 검색 수행 ---")
        return self.retriever.invoke(question)

    def find_self_query_documents(self, question):
        print(f"--- INFO: '{question}'에 대한 지능형 메타데이터 검색 수행 ---")
        return self.self_query_retriever.invoke(question)
    
    def find_contexts(self, docs):
        return [(doc.page_content + '\n' + json.dumps(doc.metadata, ensure_ascii=False)) for doc in docs]


    def get_recent_history(self, history, window_size=10):
        return history[-window_size:] if history else []

    def _create_rephrasing_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("history"),
            ("user", "{input}"),
            ("user", "주어진 대화 기록을 고려하여, 후속 질문을 검색에 용이한 독립적인 질문으로 재구성해주세요.")
        ])
        return prompt | self.llm | StrOutputParser()

    def get_hybrid_retrieved_documents(self, x: dict) -> List[Document]:
        """원본 질문과 재구성된 질문 모두로 검색하여 결과를 합치고 중복을 제거합니다."""
        original_question = x["input"]
        rephrased_question = x["rephrased_question"]

        print(f"--- INFO (Hybrid Retrieval): 원본 질문 '{original_question}'으로 검색 ---")
        docs_raw = self.find_documents(original_question)
        
        print(f"--- INFO (Hybrid Retrieval): 재구성된 질문 '{rephrased_question}'으로 검색 ---")
        docs_rephrased = self.find_documents(rephrased_question)

        print(f"--- INFO (Hybrid Retrieval + Self query retriver): 재구성된 질문 '{rephrased_question}'으로 검색 ---")
        self_query_docs_rephrased = self.find_self_query_documents(rephrased_question)

        # 두 문서 리스트를 합칩니다.
        combined_docs = docs_raw + docs_rephrased + self_query_docs_rephrased

        # 중복된 문서를 제거합니다. page_content를 기준으로 고유성을 확인합니다.
        unique_docs = {}
        for doc in combined_docs:
            unique_docs[doc.page_content] = doc
        
        dedup_docs = list(unique_docs.values())
        final_docs = sorted(dedup_docs, key=lambda d: d.metadata.get('score', 0), reverse=True)[:5]
        print(f"--- INFO (Hybrid Retrieval): 총 {len(combined_docs)}개 문서를 검색, 중복 제거 후 {len(final_docs)}개 문서 확보 ---")
        
        return final_docs
    
    def _deterministic_router(self, x: dict) -> Optional[str]:
        query = x['input'].lower()
        if any (keyword in query for keyword in ["요약", "요약해줘", "정리해줘", "브리핑"]):
            return "summarization"
        if any(keyword in query for keyword in ["추천", "비슷한 사업", "유사한 사업", "추천해줘"]):
            return "recommendation"
        if any(keyword in query for keyword in ["비교", "차이점", "대조"]):
            return "comparison"
        return None

    def create_router_chain(self):
        # 라우터 프롬프트: 쿼리 의도 분류
        rephrasing_chain = self._create_rephrasing_chain()
        route_prompt = ChatPromptTemplate.from_template(
            """당신은 사용자의 질문 의도를 정확하게 분석하여 5개의 카테고리 중 하나로 분류하는 전문가입니다.
            **'원본 질문'을 통해 사용자의 최종 목표(요약, 비교, 추천 등)를 파악하세요.**
            **오직 아래 5개의 카테고리 이름 중 하나만! 답변해야 합니다.** 다른 설명은 절대 추가하지 마세요.

            # 카테고리 목록:
            `metadata_search`: 특정 조건(공고 번호, 공고 차수, 사업명, 사업 금액, 발주 기관, 공개 일자, 입찰 참여 시작일, 입찰 참여 마감일)으로 문서를 찾아달라는 요청.
            `summarization`: 문서의 전체 내용, 특정 부분, 또는 핵심 요구사항 등을 요약해달라는 요청.
            `comparison`: 두 개 이상의 RFP 문서를 비교하거나, 한 문서 내의 두 가지 항목을 비교/대조해달라는 요청.
            `recommendation`: 특정 사업과 유사한 다른 사업을 추천해달라는 요청.
            `default_qa`: 특정 RFP 문서 내용에 대한 구체적인 세부 정보를 묻는 질문. (위 4가지에 해당하지 않는 모든 질문)

            # 분류 기준:
            - "사업 찾아줘", "목록 알려줘" -> `metadata_search`
            - "요약해줘", "브리핑해줘", "정리해줘" -> `summarization`
            - "비교해줘", "~랑 ~의 차이점 알려줘" -> `comparison`
            - "추천해줘", "비슷한 사업 찾아줘" -> `recommendation`
            - 그 외 특정 정보 질문 (e.g., "평가 방식은 뭐야?", "유지보수 기간 알려줘") -> `default_qa`

            # --- 분석할 질문 ---
            # 원본 질문: {input}
            # 검색용 질문: {rephrased_question}
            # --------------------

            # 분류 예시 (Few-shot Examples):
            질문: "국민연금공단 이러닝 사업의 예산이 얼마인가요?"
            분류: metadata_search

            질문: "2024년 10월 이후에 마감되는 모든 사업 목록을 찾아줘."
            분류: metadata_search

            질문: "부산관광공사 사업의 핵심 과업을 요약해줘."
            분류: summarization

            질문: "전체 RFP 내용에 대해 간략하게 브리핑해줘."
            분류: summarization

            질문: "고려대학교 포털 사업과 국민연금 이러닝 사업의 사업 기간을 비교해줘."
            분류: comparison
            
            질문: "부산관광공사 RFP에서, 제안서 평가 방식과 기술 협상 방식의 차이점을 설명해줘."
            분류: comparison

            질문: "AI 기반 콜센터 구축과 비슷한 사업을 찾아 추천해줄래?"
            분류: recommendation
            
            질문: "이러닝 시스템 말고 다른 교육 관련 사업도 있으면 알려줘."
            분류: recommendation
            
            질문: "국민연금공단 RFP에서, 제안서 평가는 어떤 방식으로 진행되나요?"
            분류: default_qa

            질문: "이 사업의 무상 유지보수 기간은 얼마나 되나요?"
            분류: default_qa

            # 이제 이 질문을 분류하세요:
            질문: {rephrased_question}
            분류 결과 (카테고리 이름만 출력):"""
        )

        llm_route_chain = route_prompt | self.llm | StrOutputParser()
        
        # 3단계: 각 의도에 맞는 전문화된 체인 정의
        metadata_search_chain = self._create_metadata_search_chain()
        summarization_chain = self._create_summarization_chain()
        comparison_chain = self._create_comparison_chain()
        recommendation_chain = self._create_recommendation_chain()
        default_qa_chain = self._create_default_qa_chain()

        # 전체 파이프라인 결합 - 안정적인 데이터 흐름 보장
        def log_and_pass_through(data):
            classification = data.get('classification', 'N/A')
            
            # 문자열 또는 dict 처리
            if isinstance(classification, dict):
                classification_str = classification.get('classification', 'N/A')
            else:  # str 또는 그 외
                classification_str = classification

            # 안전하게 strip
            classification_str = classification_str.strip() if isinstance(classification_str, str) else 'N/A'

            print(f"✅ 라우터 분류 결과: {classification_str}")
            # 문자열 형태로 덮어쓰기
            data['classification'] = classification_str
            return data

        routing_branch = RunnableBranch(
            (lambda x: x["deterministic_classification"] is not None,
             RunnablePassthrough.assign(classification=lambda x: x["deterministic_classification"])),
             llm_route_chain
        )

        # ✅ 전체 파이프라인
        full_chain = (
            RunnablePassthrough.assign(
                history=lambda x: self.get_recent_history(x.get("history", []))  # 답변단계에서만 쓰기 위해 유지
            )
            .assign(rephrased_question=rephrasing_chain)
            .assign(deterministic_classification=lambda x: self._deterministic_router(x))  # 결정론적 라우팅 추가
            # retriever에는 원문 input 그대로 사용
            .assign(classification=routing_branch)
            | RunnableLambda(log_and_pass_through)
            | RunnableBranch(
                (lambda x: "metadata_search" in x.get("classification", ""), metadata_search_chain),
                (lambda x: "summarization" in x.get("classification", ""), summarization_chain),
                (lambda x: "comparison" in x.get("classification", ""), comparison_chain),
                (lambda x: "recommendation" in x.get("classification", ""), recommendation_chain),
                lambda x: default_qa_chain  # 기본값
            )
        ).with_config({"callbacks": [self.tracer]})
        
        return full_chain
        


    def _create_metadata_search_chain(self):
        """## 메타데이터 검색 체인 (Route 1)"""
        def get_context(docs) -> str:
            if not docs:
                print("--- WARNING (metadata_search): 관련된 문서를 찾지 못했습니다. ---")
                return "관련된 문서를 찾을 수 없습니다."
            context_parts = []
            for doc in docs:
                metadata_str = json.dumps(doc.metadata, ensure_ascii=False, indent=2)
                part = (f"---\n[문서 정보]:\n{metadata_str}\n\n[문서 본문]:\n{doc.page_content}\n---")
                context_parts.append(part)
            return "\n\n".join(context_parts)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "당신은 대한민국 B2G(정부 대상 사업) RFP(제안요청서) 데이터베이스 검색 전문가 비디(Bidy)입니다.\n"
             "당신은 주어진 [검색 결과] 목록을 바탕으로 사용자의 [질문]에 대한 답변을 생성해야 합니다.\n\n"
             "**작업 지침:**\n"
             "1. **검색 결과 확인:** [검색 결과]에 내용이 있는지 확인합니다.\n"
             "2. **결과 기반 답변:**\n"
             "   - **결과가 있을 경우:** \"요청하신 조건에 맞는 사업 목록입니다.\"라고 서두를 시작한 뒤, [검색 결과]에 있는 사업 목록을 빠짐없이, 순서대로 제시하세요. 각 사업 정보 앞에 번호(예: 1., 2.)를 붙여주세요.\n"
             "   - **결과가 없을 경우:** \"요청하신 조건에 맞는 사업을 찾을 수 없었습니다.\"라고만 답변하세요. 다른 말을 덧붙이지 마세요.\n"
             "3. **정보 추가 금지:** [검색 결과]에 없는 내용은 절대로 언급해서는 안 됩니다."),
            MessagesPlaceholder("history"),  # ✅ 답변 단계에서만 history 반영
            ("human", "[질문]: {input}\n\n[컨텍스트]:\n{context}")
        ])
        
        return (
            RunnablePassthrough.assign(context=RunnableLambda(lambda x:get_context(self.get_hybrid_retrieved_documents(x))))
            | prompt | self.llm | StrOutputParser()
        )
        

    def _create_default_qa_chain(self):
        """## RFP 관련 기본 QA 체인 (Route 2, Fallback Route) - 메타데이터 참조 """
        
        def get_context(x: dict) -> str:
            # 함수가 딕셔너리 'x'를 받도록 하여 'input' 키에 접근하도록 통일합니다.
            question = x["input"]
            print(f"--- INFO (Default QA): '{question}'에 대한 문서 검색 수행 ---")
            docs = self.get_hybrid_retrieved_documents(x) # self.find_documents(question)
            if not docs:
                print("--- WARNING (Default QA): 관련된 문서를 찾지 못했습니다. ---")
                return "관련된 문서를 찾을 수 없습니다."
            retrieved_titles = [doc.metadata.get('project_title', '제목 없음') for doc in docs]
            print(f"--- INFO (Default QA): 검색된 문서 목록: {list(set(retrieved_titles))} ---")
            context_parts = []
            for doc in docs:
                metadata_str = json.dumps(doc.metadata, ensure_ascii=False, indent=2)
                part = (f"---\n[문서 정보]:\n{metadata_str}\n\n[문서 본문]:\n{doc.page_content}\n---")
                context_parts.append(part)
            return "\n\n".join(context_parts)

        # LLM에게 역할을 부여하고, 메타데이터 활용법을 명시적으로 지시하는 프롬프트
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "당신은 대한민국 B2G(정부 대상 사업) RFP(제안요청서) 분석을 전문으로 하는 AI 컨설턴트 비디(Bidy)입니다.\n"
                "당신은 주어진 [컨텍스트]에서 사용자의 [질문]에 대한 정답을 찾는 정보 추출 전문가입니다.\n\n"
                "**작업 절차:**\n"
                "1. **질문 분석:** 사용자의 [질문] 의도를 명확히 파악합니다.\n"
                "2. **정보 탐색:** [컨텍스트] 전체를 꼼꼼히 읽고, 질문에 답할 수 있는 **정확한 근거 문장이나 구절**을 찾습니다.\n"
                "3. **답변 생성:** 찾은 근거를 바탕으로, 질문에 대해 명확하고 간결하게 답변합니다. 답변은 항상 근거가 된 문서의 'project_title'을 먼저 언급하며 시작해야 합니다.\n\n"
                "**답변 스타일 가이드:**\n"
                "- 핵심 내용을 먼저 말하고, 필요시 부가 설명을 덧붙이는 **두괄식**으로 답변해주세요.\n"
                "- 가능한 경우, 정보를 **불렛 포인트(•)**나 **번호 매기기**를 사용하여 구조화해주세요.\n\n"
                "**매우 중요한 규칙:**\n"
                "- **근거 기반 답변:** 모든 답변은 반드시 [컨텍스트]에 기반해야 합니다. 당신의 사전 지식을 사용해서는 안 됩니다.\n"
                "- **출처 명시:** 답변의 시작 부분에 반드시 어떤 문서에서 정보를 찾았는지 명시하세요. (예: '「2024년 이러닝시스템 운영 용역」 문서에 따르면...')\n"
                "- **정보 부재 시:** [컨텍스트]를 여러 번 확인했음에도 답변의 근거를 정말 찾을 수 없을 때만, '제공된 문서에서는 질문에 대한 명확한 정보를 찾을 수 없었습니다.'라고 답변하세요."
            ),
            MessagesPlaceholder("history"),  # ✅ 답변 단계에서만 history 반영
            ("human", "[질문]: {input}\n\n[컨텍스트]:\n{context}")
        ])

        return (
            {
                "context": get_context,
                "input": lambda x: x["input"],
                "history": lambda x: x.get("history", []),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )



    def _create_summarization_chain(self):
        """## 정보 요약 체인 (Route 3)"""
        
        def get_documents(x: dict) -> List[Document]:
            question = x['input']
            print(f"--- INFO (Summarization): '{question}'에 대한 문서 검색 수행 ---")
            docs = self.get_hybrid_retrieved_documents(x) # self.find_documents(question)
            
            if not docs:
                print("--- WARNING (Summarization): 관련된 문서를 찾지 못했습니다. ---")
                return [] # 빈 리스트를 반환하여 다음 단계가 정상적으로 처리되도록 함

            retrieved_titles = [doc.metadata.get('project_title', '제목 없음') for doc in docs]
            print(f"--- INFO (Summarization): 검색된 문서 목록: {list(set(retrieved_titles))} ---")
            
            formatted_docs = []
            for doc in docs:
                metadata_str = json.dumps(doc.metadata, ensure_ascii=False, indent=2)
                combined_content = (
                    f"[문서 정보]:\n{metadata_str}\n\n"
                    f"[문서 본문]:\n{doc.page_content}"
                )
                formatted_docs.append(Document(page_content=combined_content))
            return formatted_docs

        # Map 단계
        map_prompt = ChatPromptTemplate.from_template(
            "당신은 B2G 사업 전문 컨설턴트 비디(Bidy)입니다. 주어진 [문서 정보]와 [문서 본문]을 모두 참고하여, 제안 결정에 영향을 미칠 수 있는 다음 핵심 정보들을 항목별로 요약해 주십시오.\n\n"
            "- **핵심 과업/요구사항:** (기술, 기능, 보안 등)\n"
            "- **예산/기간:** (금액, 계약 기간 등)\n"
            "- **일정:** (제안 마감일, 평가일 등)\n"
            "- **평가방식/참여조건:** (기술/가격 배점, 필수 자격 등)\n\n"
            "--- 문서 내용 ---\n"
            "{context}\n"
            "--- 끝 ---\n\n"
            "항목별 핵심 정보 요약:"
        )
        # map_prompt에 전달
        map_chain = {"context": lambda doc: doc.page_content} | map_prompt | self.llm | StrOutputParser()

        # Reduce 
        reduce_prompt = ChatPromptTemplate.from_messages([
            ("system",
            "당신은 B2G 사업 수주 전략을 수립하는 수석 컨설턴트 비디(Bidy)입니다. "
            "아래에 흩어져 있는 정보들을 종합하여, 의사결정을 위한 최종 '사업 요약 브리핑'을 작성해 주십시오.\n\n"
            "**브리핑 작성 가이드라인:**\n"
            "1. **핵심 요약 (Executive Summary):** 가장 먼저 사업명, 발주기관, 예산, 기간, 핵심 기술/과업을 한두 문장으로 요약하여 제시하세요.\n"
            "2. **본문:** '사업 목표', '주요 과업 범위', '예산 및 기간', '제안 시 주요 고려사항(평가방식, 참여자격, 특이사항 등)' 순서로 구조화하여 상세히 설명하세요.\n"
            "3. **확인 필요한 정보:** 만약 예산, 기간 등 **의사결정에 필수적인 정보가 누락되었다면, 반드시 '※ 확인 필요한 핵심 정보' 항목을 만들어 명시**해야 합니다."
            ),
            # ✅ 여기서 과거 대화 이력을 반영
            MessagesPlaceholder("history"),
            ("human",
            "--- 부분 정보 목록 ---\n"
            "{context}\n"
            "--- 끝 ---\n\n"
            "최종 사업 요약 브리핑:")
        ])
        
        # RunnablePassthrough.assign을 사용하여 파이프라인 전반에 걸쳐 'input', 'history' 등의
        # 원본 데이터를 유지하면서 새로운 키('docs', 'summaries', 'context')를 추가하는 방식
        
        # 1. 문서 검색 결과를 'docs' 키에 할당 (원본 데이터 유지)
        summarization_pipeline = RunnablePassthrough.assign(docs=get_documents)
        
        # 2. 'docs'에 대해 map_chain을 실행하고, 결과를 'summaries' 키에 할당
        summarization_pipeline = summarization_pipeline.assign(
            summaries=lambda x: map_chain.map().invoke(x['docs'])
        )

        # 3. 'summaries'를 하나의 문자열로 합쳐 'context' 키에 할당
        def combine_summaries(x: dict) -> str:
            summaries = x['summaries']
            # 검색된 문서가 없는 경우의 처리
            if len(summaries) == 1 and summaries[0] == "NO_DOCS_FOUND":
                return "요약을 위한 관련 문서를 찾지 못했습니다."
            return "\n\n---\n\n".join(summaries)

        summarization_pipeline = summarization_pipeline.assign(
            context=combine_summaries
        )

        return summarization_pipeline | reduce_prompt | self.llm | StrOutputParser()

    
    def _create_comparison_chain(self):
        """## 비교 분석 체인 (단일/다중 문서 처리) (Route 4)"""
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

                사용자 질문: {input}
                JSON 출력: """
        )
        triage_chain = triage_prompt | self.llm | triage_parser

        # 2-1: 다중 문서 비교 로직
        def get_context_for_multi_doc(item_name: str) -> str:
            if not item_name: return "비교 대상이 질문에서 추출되지 않았습니다."
            print(f"--- INFO (Multi-Doc): '{item_name}'에 대한 지능형 문서 검색 수행 ---")
            docs = self.find_self_query_documents(item_name)
            if not docs: return f"'{item_name}' 정보를 찾을 수 없습니다."
            return "\n\n".join(self.find_contexts(docs))

        multi_doc_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "당신은 두 B2G 사업의 전문 비교 분석가 비디(Bidy)입니다. 각 사업의 정보를 바탕으로, 사용자가 요청한 [비교 기준]에 따라 명확하게 차이점과 공통점을 설명해야 합니다. 표(마크다운 테이블)를 사용하여 결과를 시각적으로 정리해주세요."),
            MessagesPlaceholder("history"),  # ✅ 답변 생성에서만 history 반영
            ("human",
            "**비교 기준:** {criteria}\n\n"
            "**[사업 A: {item_A}]**\n{context_A}\n\n"
            "**[사업 B: {item_B}]**\n{context_B}")
        ])
        
        multi_doc_chain = RunnablePassthrough.assign(
            item_A=lambda x: x["triage_result"].get("item_A", "알 수 없는 항목 A"),
            item_B=lambda x: x["triage_result"].get("item_B", "알 수 없는 항목 B"),
            criteria=lambda x: x["triage_result"].get("criteria", "전반적인 특징"),
            context_A=lambda x: get_context_for_multi_doc(x["triage_result"].get("item_A")),
            context_B=lambda x: get_context_for_multi_doc(x["triage_result"].get("item_B"))
        ) | multi_doc_prompt | self.llm | StrOutputParser()
        
        # 2-2: 단일 문서 비교 로직
        def retrieve_and_extract_for_single_doc(input_dict: dict) -> dict:
            triage_result = input_dict.get("triage_result", {})
            base_doc_name = triage_result.get("base_document")
            topic_A = triage_result.get("topic_A")
            topic_B = triage_result.get("topic_B")
            history = input_dict.get("history", [])
            
            if not all([base_doc_name, topic_A, topic_B]):
                print("--- WARNING (Single-Doc): LLM이 비교 대상을 정확히 추출하지 못했습니다. ---")
                return {
                    "extracted_snippets": "비교 대상을 질문에서 정확히 추출하지 못해 비교를 수행할 수 없습니다.",
                    "base_document": base_doc_name or "알 수 없음",
                    "topic_A": topic_A or "알 수 없음",
                    "topic_B": topic_B or "알 수 없음",
                    "criteria": triage_result.get("criteria", "알 수 없음"),
                    "history": history
                }
                
            print(f"--- INFO (Single-Doc): 기준 문서 '{base_doc_name}'의 컨텍스트 검색 ---")
            base_docs = self.find_self_query_documents(base_doc_name)
            if not base_docs: return { "extracted_snippets": f"기준 문서 '{base_doc_name}'를 찾을 수 없습니다.", "history": history, **triage_result}            
            local_context = "\n\n".join(self.find_contexts(base_docs))
            
            print(f"--- INFO (Single-Doc): '{topic_A}'와 '{topic_B}'에 대한 정보 추출 ---")
            extraction_prompt = ChatPromptTemplate.from_template("문서 내용에서 '{topic_A}'와 '{topic_B}'에 대한 부분을 각각 찾아서 요약해줘.\n---\n{context}\n---")
            snippet_extractor = extraction_prompt | self.llm | StrOutputParser()
            extracted_snippets = snippet_extractor.invoke({"context": local_context, "topic_A": topic_A, "topic_B": topic_B})
            
            return {"extracted_snippets": extracted_snippets, "history": history, **triage_result}

        single_doc_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "당신은 한 문서 내의 두 가지 주제를 명확하게 비교 분석하는 전문가 비디(Bidy)입니다. 주어진 정보를 바탕으로 [비교 기준]에 따라 두 주제의 공통점과 차이점을 설명해야 합니다. 마크다운 테이블을 사용해 결과를 보기 쉽게 정리해주세요."),
            MessagesPlaceholder("history"),  # ✅ 답변 생성에서만 history 반영
            ("human", "**기준:** {criteria}\n\n{extracted_snippets}")
        ])
        single_doc_chain = RunnableLambda(retrieve_and_extract_for_single_doc) | single_doc_prompt | self.llm | StrOutputParser()

        branch = RunnableBranch(
            (lambda x: x.get("triage_result", {}).get("type") == "single_document", single_doc_chain),
            (lambda x: x.get("triage_result", {}).get("type") == "multi_document", multi_doc_chain),
            lambda x: {"extracted_snippets": "비교 유형을 식별할 수 없습니다."}
        )
        return (
            RunnablePassthrough.assign(
                triage_result=triage_chain, 
                history=lambda x: x.get("history", []) 
            ) | branch
        )


    def _create_recommendation_chain(self):
        """## 유사 사업 추천 체인 (Route 5)"""

        # Map 단계 (Query Expansion):
        query_expansion_prompt = ChatPromptTemplate.from_template(
            "당신은 B2G 사업 검색 전문가입니다. 사용자의 요청을 '핵심 기술', '사업 분야', '프로젝트 유형'의 관점에서 분석하여, "
            "벡터 검색에 가장 효과적인 검색 키워드 3개를 쉼표(,)로 구분하여 생성하세요.\n\n"
            "사용자 요청: {input}\n"
            "검색 키워드:"
        )
        
        query_expansion_chain = query_expansion_prompt | self.llm | StrOutputParser()

        
        def format_docs(docs: List[Document]) -> str:
            # 검색 결과가 없을 경우를 대비한 방어 코드
            if not docs:
                return "추천할 만한 유사 사업을 찾지 못했습니다."
            return "\n\n".join(
                f"### 사업명: {doc.metadata.get('project_title', '제목 없음')}\n"
                f"공고번호: {doc.metadata.get('rfp_number', '미상')}\n"
                f"요약:\n{doc.metadata.get('summary', doc.page_content)}"
                for doc in docs
            )
            
        # Reduce 단계 (Recommendation)
        recommendation_prompt = ChatPromptTemplate.from_messages([
            ("system",
            "당신은 B2G 사업 분석가 비디(Bidy)이며, '검색된 유사 사업 목록'을 바탕으로 사용자에게 맞춤형 사업을 추천하는 전문가입니다.\n\n"
             "**작업 지침:**\n"
             "1. **사용자 요청 분석:** '사용자 원본 요청'을 파악하여 어떤 종류의 사업을 원하는지 이해합니다.\n"
             "2. **유사도 판단:** '검색된 유사 사업 목록'의 각 사업이 사용자 요청과 얼마나 유사한지 비교 분석합니다.\n"
             "3. **추천 목록 생성:**\n"
             "   - **결과가 있을 경우:** 유사도가 가장 높다고 판단되는 사업을 **최대 3개까지** 추천합니다. 추천 목록은 **번호(1., 2., 3.)**를 붙여주세요.\n"
             "   - 각 추천 항목에는 반드시 **'사업명'**과 **'발주기관'**을 포함해야 합니다.\n"
             "   - **가장 중요한 것은, '어떤 점에서 유사한지' 구체적인 이유와 근거를 명확하게 설명**해야 합니다.\n"
             "   - **결과가 없을 경우:** '검색된 유사 사업 목록'에 '추천할 만한 유사 사업을 찾지 못했습니다.'라는 내용이 있다면, \"요청하신 내용과 유사한 사업을 찾을 수 없었습니다.\"라고만 답변하세요.\n\n"
             "**매우 중요한 규칙:**\n"
             "- **근거 기반 추천:** 당신의 모든 추천은 반드시 '검색된 유사 사업 목록'에 있는 정보에만 근거해야 합니다."
            "--- (규칙 끝) ---"
            ),
            # ✅ 여기서 멀티턴 맥락(history) 반영
            MessagesPlaceholder("history"),
            ("human", 
             "**사용자 원본 요청:**\n{input}\n\n"
             "**검색된 유사 사업 목록:**\n{context}\n\n"
             "**추천 목록 (위 지침에 따라 작성):**")
        ])
        
        recommendation_pipeline = RunnablePassthrough.assign(
            expanded_query=query_expansion_chain # 'input'을 받아 'expanded_query' 생성
        ).assign(
            # 생성된 'expanded_query'를 사용해 MMR 검색 후 'context' 생성
            context=lambda x: format_docs(self.get_hybrid_retrieved_documents(x))
        )
        
        # 최종적으로 'input', 'history', 'expanded_query', 'context'가 모두 포함된 딕셔너리가 recommendation_prompt로 전달.
        return recommendation_pipeline | recommendation_prompt | self.llm | StrOutputParser()