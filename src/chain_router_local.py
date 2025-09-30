# chain_router_local.py
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain.chat_models import ChatLocalLLM  # 변경: 로컬 LLM
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

    def get_recent_history(self, history, window_size=5):
        return history[-window_size:] if history else []

    def _create_rephrasing_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("history"),
            ("user", "{input}"),
            ("user", "주어진 대화 기록을 고려하여, 후속 질문을 검색에 용이한 독립적인 질문으로 재구성해주세요.")
        ])
        return prompt | self.llm | StrOutputParser()

    def get_hybrid_retrieved_documents(self, x: dict) -> List[Document]:
        original_question = x["input"]
        rephrased_question = x["rephrased_question"]

        print(f"--- INFO (Hybrid Retrieval): 원본 질문 '{original_question}'으로 검색 ---")
        docs_raw = self.find_documents(original_question)
        
        print(f"--- INFO (Hybrid Retrieval): 재구성된 질문 '{rephrased_question}'으로 검색 ---")
        docs_rephrased = self.find_documents(rephrased_question)

        combined_docs = docs_raw + docs_rephrased
        unique_docs = {doc.page_content: doc for doc in combined_docs}
        final_docs = list(unique_docs.values())
        print(f"--- INFO (Hybrid Retrieval): 총 {len(combined_docs)}개 문서를 검색, 중복 제거 후 {len(final_docs)}개 문서 확보 ---")
        
        return final_docs

    def create_router_chain(self):
        rephrasing_chain = self._create_rephrasing_chain()
        route_prompt = ChatPromptTemplate.from_template(
            """사용자의 질문을 분석하여, 질문의 의도를 다음 5가지 카테고리 중 하나로 분류하세요.
            `metadata_search`, `summarization`, `comparison`, `recommendation`, `default_qa`
            
            질문: {rephrased_question}
            분류:"""
        )

        route_chain = route_prompt | self.llm | StrOutputParser()

        # 전문 체인 생성
        metadata_search_chain = self._create_metadata_search_chain()
        summarization_chain = self._create_summarization_chain()
        comparison_chain = self._create_comparison_chain()
        recommendation_chain = self._create_recommendation_chain()
        default_qa_chain = self._create_default_qa_chain()

        def log_and_pass_through(data):
            classification = data.get('classification', 'N/A')
            if isinstance(classification, dict):
                classification_str = classification.get('classification', 'N/A')
            else:
                classification_str = classification
            classification_str = classification_str.strip() if isinstance(classification_str, str) else 'N/A'
            print(f"✅ 라우터 분류 결과: {classification_str}")
            data['classification'] = classification_str
            return data

        full_chain = (
            RunnablePassthrough.assign(
                history=lambda x: self.get_recent_history(x.get("history", []))
            )
            .assign(rephrased_question=rephrasing_chain)
            .assign(classification=route_chain)
            | RunnableLambda(log_and_pass_through)
            | RunnableBranch(
                (lambda x: "metadata_search" in x.get("classification", ""), metadata_search_chain),
                (lambda x: "summarization" in x.get("classification", ""), summarization_chain),
                (lambda x: "comparison" in x.get("classification", ""), comparison_chain),
                (lambda x: "recommendation" in x.get("classification", ""), recommendation_chain),
                lambda x: default_qa_chain
            )
        ).with_config({"callbacks": [self.tracer]})
        
        return full_chain

    # --- Route Chains ---
    def _create_metadata_search_chain(self):
        def get_context(docs) -> str:
            if not docs:
                return "관련된 문서를 찾을 수 없습니다."
            return "\n\n".join([f"[문서 정보]:\n{json.dumps(doc.metadata, ensure_ascii=False, indent=2)}\n[문서 본문]:\n{doc.page_content}" for doc in docs])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "친절한 AI 어시스턴트: 특정 조건으로 문서를 찾아주세요."),
            MessagesPlaceholder("history"),
            ("human", "[질문]: {input}\n\n[컨텍스트]:\n{context}")
        ])
        return RunnablePassthrough.assign(context=RunnableLambda(lambda x:get_context(self.find_self_query_documents(x['input'])))) | prompt | self.llm | StrOutputParser()

    def _create_default_qa_chain(self):
        def get_context(x: dict) -> str:
            question = x["input"]
            docs = self.get_hybrid_retrieved_documents(x)
            if not docs:
                return "관련된 문서를 찾을 수 없습니다."
            return "\n\n".join([f"[문서 정보]:\n{json.dumps(doc.metadata, ensure_ascii=False, indent=2)}\n[문서 본문]:\n{doc.page_content}" for doc in docs])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "RFP 분석 전문 AI."),
            MessagesPlaceholder("history"),
            ("human", "[질문]: {input}\n\n[컨텍스트]:\n{context}")
        ])
        return RunnablePassthrough.assign(
            context=get_context,
            input=lambda x: x["input"],
            history=lambda x: x.get("history", [])
        ) | prompt | self.llm | StrOutputParser()

    def _create_summarization_chain(self):
        def get_documents(x: dict) -> List[Document]:
            docs = self.find_documents(x['input'])
            return [Document(page_content=f"[문서 정보]:\n{json.dumps(doc.metadata, ensure_ascii=False)}\n[문서 본문]:\n{doc.page_content}") for doc in docs] if docs else []
        
        map_prompt = ChatPromptTemplate.from_template(
            "주어진 [문서 정보]와 [문서 본문]을 참고하여 핵심 정보를 항목별로 요약해 주세요.\n{context}"
        )
        map_chain = {"context": lambda doc: doc.page_content} | map_prompt | self.llm | StrOutputParser()
        
        reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", "최종 요약 브리핑 작성"),
            MessagesPlaceholder("history"),
            ("human", "--- 부분 정보 목록 ---\n{context}\n--- 끝 ---\n최종 사업 요약 브리핑:")
        ])
        
        summarization_pipeline = RunnablePassthrough.assign(docs=get_documents)
        summarization_pipeline = summarization_pipeline.assign(summaries=lambda x: map_chain.map().invoke(x['docs']))
        summarization_pipeline = summarization_pipeline.assign(context=lambda x: "\n\n---\n\n".join(x['summaries']))
        return summarization_pipeline | reduce_prompt | self.llm | StrOutputParser()

    def _create_comparison_chain(self):
        triage_parser = JsonOutputParser()
        triage_prompt = ChatPromptTemplate.from_template(
            "질문을 분석하여 'multi_document' 또는 'single_document' 유형으로 분류, 관련 정보를 JSON으로 추출하세요.\n질문: {input}\nJSON 출력:"
        )
        triage_chain = triage_prompt | self.llm | triage_parser

        # 단순화: multi/single branch
        branch = RunnableBranch(
            (lambda x: x.get("triage_result", {}).get("type") == "single_document", RunnableLambda(lambda x: x)),
            (lambda x: x.get("triage_result", {}).get("type") == "multi_document", RunnableLambda(lambda x: x)),
            lambda x: {"extracted_snippets": "비교 유형을 식별할 수 없습니다."}
        )
        return RunnablePassthrough.assign(triage_result=triage_chain, history=lambda x: x.get("history", [])) | branch

    def _create_recommendation_chain(self):
        query_expansion_prompt = ChatPromptTemplate.from_template(
            "사용자의 요청: {input}\n검색 키워드 3개 생성:"
        )
        query_expansion_chain = query_expansion_prompt | self.llm | StrOutputParser()

        def format_docs(docs: List[Document]) -> str:
            if not docs:
                return "추천할 만한 유사 사업을 찾지 못했습니다."
            return "\n\n".join(
                f"### 사업명: {doc.metadata.get('project_title', '제목 없음')}\n공고번호: {doc.metadata.get('rfp_number', '미상')}\n요약:\n{doc.metadata.get('summary', doc.page_content)}"
                for doc in docs
            )
        
        recommendation_prompt = ChatPromptTemplate.from_messages([
            ("system", "유사 사업 추천"),
            MessagesPlaceholder("history"),
            ("human", "**추천 목록:**")
        ])
        
        mmr_retriever = self.vectorstore.as_retriever(search_type="mmr")
        recommendation_pipeline = RunnablePassthrough.assign(expanded_query=query_expansion_chain)
        recommendation_pipeline = recommendation_pipeline.assign(context=lambda x: format_docs(mmr_retriever.get_relevant_documents(x["expanded_query"])))
        return recommendation_pipeline | recommendation_prompt | self.llm | StrOutputParser()
