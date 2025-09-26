import os
import json
from typing import List, Dict
from langsmith import Client
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
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
        self.langsmith_client = Client(
            api_key=LangSmithConfig.LANGCHAIN_API_KEY,
            api_url=LangSmithConfig.LANGCHAIN_ENDPOINT
        )
        
        self.tracer = LangChainTracer(
            project_name=LangSmithConfig.LANGCHAIN_PROJECT
        )

        self.initialize_components()
        self.history = self.load_history()
        router = ChainRouter(llm=self.llm,  retriever=self.retriever, vectorstore=self.vectorstore, tracer=self.tracer,
            find_documents_func=self.find_documents, find_contexts_func=self.find_contexts
        )
        self.chain = router.create_router_chain()
        #self.chain = self.create_router_chain() # self.create_chain()
        self.rag_handler = RAGCallbackHandler()

    def initialize_components(self):
        """Initializes the core components like LLM, embeddings, and retriever."""
        self.embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            persist_directory=Config.VECTOR_DB_PATH,
            embedding_function=self.embeddings,
            collection_name=Config.RFP_COLLECTION
        )
        self.llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=Config.TEMPERATURE)

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
                    type="string", 
                    description="공개 일자. 공고가 공개된 날짜. YYYYMMDD 형식의 정수값입니다. 예를 들어, '2024년 1월 1일'은 20240101로 변환해야 합니다."
                ),
                AttributeInfo(
                    name="bid_start_date", 
                    type="string", 
                    description="입찰 참여 시작일. 입찰 참여가 시작되는 날짜. YYYYMMDD 형식의 정수값입니다. 예를 들어, '2023년 12월 25일'은 20231225로 변환해야 합니다."
                ),
                AttributeInfo(
                    name="bid_end_date", 
                    type="string", 
                    description="입찰 참여 마감일. 입찰 참여가 마감되는 날짜. YYYYMMDD 형식의 정수값입니다. 예를 들어, '작년'이나 '2023년 이후' 같은 표현도 YYYYMMDD 정수 형식으로 바꿔서 비교해야 합니다."
                ),
                AttributeInfo(name="summary", type="string", description="사업 요약"),
                AttributeInfo(name="filename", type="string", description="파일명")
            ],
            search_kwargs={"k": Config.TOP_K + 10},
            verbose=True
        )

        reranker_model = HuggingFaceCrossEncoder(model_name=Config.RERANK_MODEL)
        compressor = CrossEncoderReranker(model=reranker_model, top_n=Config.TOP_K)
        self.retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

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

    def find_documents(self, question):
        return self.retriever.get_relevant_documents(question)
    
    def find_contexts(self, docs):
        return [(doc.page_content + '\n' + json.dumps(doc.metadata, ensure_ascii=False)) for doc in docs]

    def ask(self, query: str, is_save=True) -> str:
        """사용자 질문에 답변합니다."""
        response = self.chain.invoke({
            "input": query, # 'question' 대신 'input' 키 사용
            "history": self.history # 'history' 대신 'chat_history' 키 사용
        })

        if isinstance(response, Dict) and 'answer' in response:
            answer = response['answer']
        else:
            answer = response
        
        self.history.append(HumanMessage(content=query))
        self.history.append(AIMessage(content=str(answer)))
        
        if is_save:
            self.save_history()

        return str(answer)