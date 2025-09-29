# chain_router.py
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
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
        return self.retriever.invoke(question)
    
    def find_contexts(self, docs):
        return [(doc.page_content + '\n' + json.dumps(doc.metadata, ensure_ascii=False)) for doc in docs]

    def get_recent_history(self, history, window_size=5):
        return history[-window_size:] if history else []

    def create_router_chain(self):
        # 컨텍스트 프롬프트: 이전 대화 맥락 유지
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 대화 맥락을 반영해서 사용자의 현재 질문을 독립적인 형태로 재구성하는 도우미야. 이전 대화 내용을 참고해서, 현재 질문이 더 명확해지도록 만들어줘."),
            MessagesPlaceholder("history"),
            ("human", "{input}")
        ])

        contextualizer_chain = contextualize_prompt | self.llm | StrOutputParser()

        # 라우터 프롬프트: 쿼리 의도 분류
        route_prompt = ChatPromptTemplate.from_template(
            """당신은 사용자의 질문 의도를 5개의 카테고리 중 하나로 완벽하게 분류하는 전문가입니다.
            주어진 질문을 분석하여 가장 적절한 카테고리 이름 하나만 답변해야 합니다.

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
            classification = data.get('classification', 'N/A').strip()
            print(f"✅ 라우터 분류 결과: {classification}")
            return data

        full_chain = (
            RunnablePassthrough.assign(
                history=lambda x: self.get_recent_history(x.get("history", []))
            )
            | RunnablePassthrough.assign(refined_query=contextualizer_chain)
            .assign(classification=lambda x: route_chain.invoke({"input": x['refined_query']}))
            | RunnableLambda(log_and_pass_through)
            | RunnableBranch(
                (lambda x: "general" in x.get("classification", ""), general_chain),
                (lambda x: "summarization" in x.get("classification", ""), summarization_chain),
                (lambda x: "comparison" in x.get("classification", ""), comparison_chain),
                (lambda x: "recommendation" in x.get("classification", ""), recommendation_chain),
                default_qa_chain  # 기본값
            )
        )
        
        return full_chain.with_config({"callbacks": [self.tracer]})
        




    def _create_general_chain(self):
        """## 일반 대화 체인 (Route 1)"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 사용자와 자유롭게 대화하는 친절한 AI 어시스턴트야. RFP 문서가 아닌 일반적인 주제에 대해 답변해줘."),
            ("human", "{refined_query}")
        ])
        return prompt | self.llm | StrOutputParser()






    def _create_default_qa_chain(self):
        """## RFP 관련 기본 QA 체인 (Route 2, Fallback Route) - 메타데이터 참조 """
        
        def get_context_with_metadata(question: str) -> str:
            print(f"--- INFO (Default QA): '{question}'에 대한 문서 검색 수행 ---")
            docs = self.find_documents(question)
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
                "당신은 대한민국 B2G(정부 대상 사업) RFP(제안요청서) 분석을 전문으로 하는 아주 똑똑하고 정확한 AI 컨설턴트입니다.\n"
                "당신은 주어진 [컨텍스트]에서 사용자의 [질문]에 대한 정답을 찾는 데 특화된 정보 추출 전문가입니다.\n\n"
                "**작업 절차:**\n"
                "1. **집중 분석:** 사용자의 [질문]을 분석한 후, [컨텍스트]에서 질문에 답할 수 있는 **정확한 문장이나 구절**을 찾습니다.\n"
                "2. **인용 및 답변:** 찾아낸 문장이나 구절을 근거로 하여, 질문에 대한 명확하고 간결한 답변을 생성합니다. 답변 시에는 근거가 된 문서의 'project_title'을 반드시 언급해야 합니다.\n\n"

                "**매우 중요한 규칙:**\n"
                "- **답변 우선:** 당신의 최우선 목표는 어떻게든 컨텍스트 내에서 답변을 찾아내는 것입니다. 정보가 있다면 절대로 '정보를 찾을 수 없다'고 답변해서는 안 됩니다.\n"
                "- **출처 제시:** 모든 답변은 반드시 [컨텍스트]에 기반해야 하며, 어떤 문서에서 정보를 찾았는지 명시해야 합니다. (예: '「2024년 이러닝시스템 운영 용역」 문서에 따르면...')\n"
                "- **정보 부재 시:** 여러 번 확인했음에도 불구하고 컨텍스트에 답변의 근거가 될 내용이 정말로 없다면, 그때서야 '제공된 문서에서는 질문에 대한 명확한 정보를 찾을 수 없었습니다.'라고 답변하세요."
            ),
            ("human", "[질문]: {question}\n\n[컨텍스트]:\n{context}")
        ])

        return ({"context": lambda x: get_context_with_metadata(x["refined_query"]), "question": lambda x: x["refined_query"]} | prompt | self.llm | StrOutputParser())






    def _create_summarization_chain(self):
        """## 정보 요약 체인 (Route 3)"""
        
        def get_documents(x):
            question = x['refined_query']
            print(f"--- INFO (Summarization): '{question}'에 대한 문서 검색 수행 ---")
            docs = self.find_documents(question)
            
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
            "당신은 B2G 사업 전문 컨설턴트입니다. 주어진 [문서 정보]와 [문서 본문]을 모두 참고하여, 제안 결정에 영향을 미칠 수 있는 다음 핵심 정보들을 항목별로 요약해 주십시오.\n\n"
            "- **핵심 과업/요구사항:** (기술, 기능, 보안 등)\n"
            "- **예산/기간:** (금액, 계약 기간 등)\n"
            "- **일정:** (제안 마감일, 평가일 등)\n"
            "- **평가방식/참여조건:** (기술/가격 배점, 필수 자격 등)\n\n"
            "--- 문서 내용 ---\n"
            "{text}\n"
            "--- 끝 ---\n\n"
            "항목별 핵심 정보 요약:"
        )
        # map_prompt에 전달
        map_chain = {"text": lambda doc: doc.page_content} | map_prompt | self.llm | StrOutputParser()

        # Reduce 
        reduce_prompt = ChatPromptTemplate.from_template(
            "당신은 B2G 사업 수주 전략을 수립하는 수석 컨설턴트입니다. 아래에 흩어져 있는 정보들을 종합하여, 의사결정을 위한 최종 '사업 요약 브리핑'을 작성해 주십시오.\n\n"
            "**브리핑 작성 가이드라인:**\n"
            "1. **핵심 요약 (Executive Summary):** 가장 먼저 사업명, 발주기관, 예산, 기간, 핵심 기술/과업을 한두 문장으로 요약하여 제시하세요.\n"
            "2. **본문:** '사업 목표', '주요 과업 범위', '예산 및 기간', '제안 시 주요 고려사항(평가방식, 참여자격, 특이사항 등)' 순서로 구조화하여 상세히 설명하세요.\n"
  
            "3. **확인 필요한 정보:** 만약 예산, 기간 등 **의사결정에 필수적인 정보가 누락되었다면, 반드시 '※ 확인 필요한 핵심 정보' 항목을 만들어 명시**해야 합니다.\n\n"
            "--- 부분 정보 목록 ---\n"
            "{text}\n"
            "--- 끝 ---\n\n"
            "최종 사업 요약 브리핑:"
        )
        # 합쳐진 부분 요약들을 text 변수에 매핑하여 reduce_prompt에 전달
        reduce_chain = {"text": lambda summaries: "\n\n---\n\n".join(summaries)} | reduce_prompt | self.llm | StrOutputParser()
        
        # map의 결과(문자열 리스트)를 reduce가 처리할 수 있는 형태(단일 문자열)로 변환하는 단계를 추가
        return (RunnableLambda(get_documents) | map_chain.map() | reduce_chain)




    
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

                사용자 질문: {question}
                JSON 출력: """
        )
        # 이제 'refined_query'를 입력으로 받습니다.
        triage_chain = {"question": lambda x: x['refined_query']} | triage_prompt | self.llm | triage_parser

        # 2-1: 다중 문서 비교 로직
        def get_context_for_multi_doc(item_name: str) -> str:
            if not item_name: return "비교 대상이 질문에서 추출되지 않았습니다."
            print(f"--- INFO (Multi-Doc): '{item_name}'에 대한 지능형 문서 검색 수행 ---")
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
            
            if not all([base_doc_name, topic_A, topic_B]):
                print("--- WARNING (Single-Doc): LLM이 비교 대상을 정확히 추출하지 못했습니다. ---")
                return {
                    "extracted_snippets": "비교 대상을 질문에서 정확히 추출하지 못해 비교를 수행할 수 없습니다.",
                    "base_document": base_doc_name or "알 수 없음",
                    "topic_A": topic_A or "알 수 없음",
                    "topic_B": topic_B or "알 수 없음",
                    "criteria": triage_result.get("criteria", "알 수 없음"),
                }
                
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
            (lambda x: x.get("triage_result", {}).get("type") == "single_document", single_doc_chain),
            (lambda x: x.get("triage_result", {}).get("type") == "multi_document", multi_doc_chain),
            (lambda x: "비교 유형을 식별할 수 없습니다.")
        )
        return RunnablePassthrough.assign(triage_result=triage_chain) | branch






    def _create_recommendation_chain(self):
        """## 유사 사업 추천 체인 (Route 5)"""

        # Map 단계 (Query Expansion):
        query_expansion_prompt = ChatPromptTemplate.from_template(
            "당신은 B2G 사업 검색 전문가입니다. 사용자의 요청을 '핵심 기술', '사업 분야', '프로젝트 유형'의 관점에서 분석하여, "
            "벡터 검색에 가장 효과적인 검색 키워드 3개를 쉼표(,)로 구분하여 생성하세요.\n\n"
            "사용자 요청: {question}\n"
            "검색 키워드:"
        )
        
        query_expansion_chain = {"question": lambda x: x['refined_query']} | query_expansion_prompt | self.llm | StrOutputParser()

        
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
        recommendation_prompt = ChatPromptTemplate.from_template(
            "당신은 '검색된 유사 사업 목록'만을 사용하여 사용자의 요청과 유사한 사업을 추천하는 B2G 사업 분석가입니다.\n\n"
            "**사용자 원본 요청:**\n{original_input}\n\n"
            "**검색된 유사 사업 목록:**\n{context}\n\n"
            "--- (매우 중요한 규칙) ---\n"
            "1. **역할 준수:** 당신의 유일한 임무는 '검색된 유사 사업 목록'에 있는 사업들을 '사용자 원본 요청'과 비교하여 추천 목록을 만드는 것입니다.\n"
            "2. **직접 답변 금지:** 절대로 '사용자 원본 요청'에 대해 당신의 자체 지식으로 직접 답변을 생성하면 안 됩니다.\n"
            "3. **근거 기반 추천:** '검색된 유사 사업 목록'을 바탕으로, 가장 유사도가 높다고 생각하는 순서대로 최대 3개의 사업을 추천하고, 어떤 점이 유사한지 명확한 근거를 제시해야 합니다.\n"
            "4. **결과 없음 처리:** 만약 '검색된 유사 사업 목록'에 '추천할 만한 유사 사업을 찾지 못했습니다.'라는 내용이 있다면, 다른 말을 덧붙이지 말고 \"요청하신 내용과 유사한 사업을 찾을 수 없었습니다.\"라고만 답변하십시오.\n"
            "--- (규칙 끝) ---\n\n"
            "**추천 목록 (위 규칙에 따라 작성):**"
        )
        
        mmr_retriever = self.vectorstore.as_retriever(search_type="mmr")

        return (
            {
                "expanded_query": query_expansion_chain,
                "original_input": lambda x: x['refined_query']
            }
            | RunnablePassthrough.assign(
                context=lambda x: format_docs(mmr_retriever.get_relevant_documents(x["expanded_query"]))
            )
            | recommendation_prompt
            | self.llm
            | StrOutputParser()
        )