# chain_router.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, chain
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from typing import List, Dict
from langchain.schema.output_parser import StrOutputParser

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
        """## 유사 사업 추천 체인 (Route 5)"""
        # 의미적 유사성 검색을 활용
        return (
            RunnablePassthrough.assign(
                docs=RunnableLambda(lambda x: self.vectorstore.as_retriever(search_type="mmr").get_relevant_documents(x['input']))
            )
            | RunnableLambda(
                lambda x: "다음은 의미적으로 유사한 사업 추천 목록입니다.\n\n" + "\n\n".join([
                    f"- {doc.metadata.get('project_title', '제목 없음')} (공고번호: {doc.metadata.get('rfp_number', '미상')})"
                    for doc in x['docs']
                ])
            )
        )
