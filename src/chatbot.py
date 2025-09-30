import os
import json
import time
import numpy as np
from typing import List, Dict
from langsmith import Client
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda, chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.cross_encoders import HuggingFaceCrossEncoder # 리랭커 임포트
from langchain.retrievers import ContextualCompressionRetriever # 압축 리트리버 임포트
from langchain.retrievers.document_compressors import CrossEncoderReranker # 크로스 인코더 압축기 임포트
from langchain.callbacks.tracers import LangChainTracer

from config import Config, LangSmithConfig
from rag_graph import RAGCallbackHandler
from chain_router import ChainRouter


class Chatbot:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.history_file = os.path.join(Config.HISTORY_PATH, f"{self.user_id}_history.json")

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # Validate LangSmith configuration
        LangSmithConfig.validate_config()
        
        # Initialize LangSmith client with explicit API key
        self.langsmith_client = Client(api_key=LangSmithConfig.LANGCHAIN_API_KEY, api_url=LangSmithConfig.LANGCHAIN_ENDPOINT)        
        self.tracer = LangChainTracer(project_name=LangSmithConfig.LANGCHAIN_PROJECT)
        self.initialize_components()
        self.history = self.load_history()
        self.router = ChainRouter(llm=self.llm,  retriever=self.retriever, self_query_retriever=self.self_query_retriever, vectorstore=self.vectorstore, tracer=self.tracer)
        self.chain = self.router.create_router_chain()
        self.rag_handler = RAGCallbackHandler()
        self.reranker_model = HuggingFaceCrossEncoder(model_name=Config.RERANK_MODEL)
        self.cross_reranker = CrossEncoderReranker(model=self.reranker_model, top_n=Config.TOP_K)

    def initialize_components(self):
        """Initializes the core components like LLM, embeddings, and retriever."""
        self.embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            persist_directory=Config.VECTOR_DB_PATH,
            embedding_function=self.embeddings,
            collection_name=Config.RFP_COLLECTION
        )
        self.llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=Config.TEMPERATURE)
        self.retriever = self.create_default_retriever()
        self.self_query_retriever = self.create_self_query_retriever()


    def create_default_retriever(self):
        base_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": Config.FIRST_TOP_K, "fetch_k": Config.FETCH_K, "lambda_mult": Config.LAMBDA_MULT}  
        )
        reranker_model = HuggingFaceCrossEncoder(model_name=Config.RERANK_MODEL)
        compressor = CrossEncoderReranker(model=reranker_model, top_n=Config.TOP_K)
        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)


    def create_self_query_retriever(self):
        base_retriever = SelfQueryRetriever.from_llm(
            self.llm,
            self.vectorstore,
            document_contents="정부 및 공공기관에서 발주하는 RFP(제안요청서)의 상세 내용. 사업 개요, 예산, 기간, 제안 조건 등을 포함함.",
            metadata_field_info=[
                AttributeInfo(name="rfp_number", type="string", description="공고 번호"),
                AttributeInfo(name="project_title", type="string", description="사업명"),
                AttributeInfo(name="budget_krw", type="integer", description="사업 금액"),
                AttributeInfo(name="agency", type="string", description="발주 기관"),
                AttributeInfo(
                    name="publish_date", 
                    type="integer", 
                    description="공개 일자. 공고가 공개된 날짜. YYYYMMDD 형식의 정수값입니다. 예를 들어, '2024년 1월 1일'은 20240101로 변환해야 합니다."
                ),
                AttributeInfo(
                    name="bid_start_date", 
                    type="integer", 
                    description="입찰 참여 시작일. 입찰 참여가 시작되는 날짜. YYYYMMDD 형식의 정수값입니다. 예를 들어, '2023년 12월 25일'은 20231225로 변환해야 합니다."
                ),
                AttributeInfo(
                    name="bid_end_date", 
                    type="integer", 
                    description="입찰 참여 마감일. 입찰 참여가 마감되는 날짜. YYYYMMDD 형식의 정수값입니다. 예를 들어, '작년'이나 '2023년 이후' 같은 표현도 YYYYMMDD 정수 형식으로 바꿔서 비교해야 합니다."
                ),
                AttributeInfo(name="summary", type="string", description="사업 요약"),
                AttributeInfo(name="filename", type="string", description="파일명")
            ],
            search_kwargs={"k": Config.FIRST_TOP_K},
            verbose=True
        )
        reranker_model = HuggingFaceCrossEncoder(model_name=Config.RERANK_MODEL)
        compressor = CrossEncoderReranker(model=reranker_model, top_n=Config.TOP_K)
        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
        

    def find_documents(self, question):
        return self.router.find_documents(question)
    
    def find_contexts(self, docs):
        return self.router.find_contexts(docs)

    def load_history(self) -> list:
        if not os.path.exists(Config.HISTORY_PATH):
            os.makedirs(Config.HISTORY_PATH)

        if os.path.exists(self.history_file):
            with open(self.history_file, "r", encoding="utf-8") as f:
                history_json = json.load(f)
                return [
                    HumanMessage(content=msg['content']) if msg['type'] == 'human'
                    else AIMessage(content=msg['content']) for msg in history_json
                ]
        return []

    def save_history(self):
        with open(self.history_file, "w", encoding="utf-8") as f:
            history_json = [
                {'type': 'human', 'content': msg.content} if isinstance(msg, HumanMessage)
                else {'type': 'ai', 'content': msg.content} for msg in self.history
            ]
            json.dump(history_json, f, ensure_ascii=False, indent=4)

    def ask(self, question: str, save_history_flag: bool = True) -> dict:
        """
        Asks the chatbot a question and returns the answer and the context used.
        
        Returns:
            A dictionary containing 'answer' and 'context_docs'.
        """
        # 추론 시간 측정
        start_time = time.time()

        inputs = {"input": question, "history": self.history}
        
        # invoke 결과 가져오기
        result = self.chain.invoke(inputs)

        # --- 🔹 result가 str인지 dict인지 확인 ---
        if isinstance(result, dict):
            answer = result.get('answer', '')
            context_docs = result.get('docs', [])
        else:
            answer = str(result)  # str이면 그대로 answer
            # context_docs를 재호출하지 않고 안전하게 빈 리스트
            context_docs = self.find_documents(question)

        # 안전하게 Document 변환
        safe_docs = []
        for doc in context_docs:
            if isinstance(doc, Document):
                safe_docs.append(doc)
            elif isinstance(doc, str):
                safe_docs.append(Document(page_content=doc, metadata={}))
            else:
                safe_docs.append(Document(page_content=str(doc), metadata={}))

        # History 업데이트
        if save_history_flag:
            self.history.extend([
                HumanMessage(content=question),
                AIMessage(content=answer),
            ])
            self.save_history()

        inference_time = time.time() - start_time
        return {
            "answer": answer,
            "context_docs": safe_docs,
            "inference_time": inference_time
        }