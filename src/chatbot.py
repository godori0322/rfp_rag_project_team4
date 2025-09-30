import os
import json
from typing import List
from langsmith import Client
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chat_models import ChatLocalLLM
from langchain.callbacks.tracers import LangChainTracer
from chain_router_local import ChainRouter  # 앞서 만든 로컬 LLM용 chain_router

from config import Config, LangSmithConfig
from rag_graph import RAGCallbackHandler


class Chatbot:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.history_file = os.path.join(Config.HISTORY_PATH, f"{self.user_id}_history.json")
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # LangSmith 설정 검증
        LangSmithConfig.validate_config()
        self.langsmith_client = Client(api_key=LangSmithConfig.LANGCHAIN_API_KEY,
                                       api_url=LangSmithConfig.LANGCHAIN_ENDPOINT)
        self.tracer = LangChainTracer(project_name=LangSmithConfig.LANGCHAIN_PROJECT)

        self.initialize_components()
        self.history = self.load_history()
        self.router = ChainRouter(
            llm=self.llm,
            retriever=self.retriever,
            self_query_retriever=self.self_query_retriever,
            vectorstore=self.vectorstore,
            tracer=self.tracer
        )
        self.chain = self.router.create_router_chain()
        self.rag_handler = RAGCallbackHandler()

    def initialize_components(self):
        """LLM, 임베딩, 벡터스토어, 리트리버 초기화"""
        from langchain_openai import OpenAIEmbeddings
        self.embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)

        self.vectorstore = Chroma(
            persist_directory=Config.VECTOR_DB_PATH,
            embedding_function=self.embeddings,
            collection_name=Config.RFP_COLLECTION
        )

        # 변경: 로컬 LLM 사용
        self.llm = ChatLocalLLM(model_path=Config.LOCAL_LLM_MODEL_PATH, temperature=Config.TEMPERATURE)

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
            document_contents="정부 및 공공기관에서 발주하는 RFP(제안요청서) 상세 내용",
            metadata_field_info=Config.METADATA_FIELD_INFO,
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
        inputs = {"input": question, "history": self.history}

        # 로컬 LLM용 chain 호출
        result = self.chain.invoke(inputs)

        # context docs는 라우터를 통해 재검색
        context_docs = self.find_documents(question)

        if save_history_flag:
            self.history.extend([HumanMessage(content=question), AIMessage(content=result)])
            self.save_history()

        return {"answer": result, "context_docs": context_docs}