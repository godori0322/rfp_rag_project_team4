import os
import json
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from config import Config, LangSmithConfig
from rag_graph import RAGCallbackHandler

class Chatbot:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.history_file = os.path.join(Config.HISTORY_PATH, f"{self.user_id}_history.json")

        load_dotenv(find_dotenv())
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
        self.chain = self.create_chain()
        self.rag_handler = RAGCallbackHandler()

    def initialize_components(self):
        self.embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        self.vectorstore = Chroma(persist_directory=Config.VECTOR_DB_PATH, embedding_function=self.embeddings, collection_name=Config.RFP_COLLECTION)
        self.llm = ChatOpenAI(
            model_name=Config.LLM_MODEL, 
            temperature=Config.TEMPERATURE,
            callbacks=[self.tracer]
        )
        self.retriever = SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.vectorstore,
            document_contents="공고 및 사업 관련 내용",
            metadata_field_info=[
                AttributeInfo(name="rfp_number", type="string", description="공고 번호"),
                AttributeInfo(name="project_title", type="string", description="사업명"),
                AttributeInfo(name="budget_krw", type="integer", description="사업 금액"),
                AttributeInfo(name="agency", type="string", description="발주 기관"),
                AttributeInfo(name="publish_date", type="date", description="공개 일자"),
                AttributeInfo(name="bid_start_date", type="date", description="입찰 참여 시작일"),
                AttributeInfo(name="bid_end_date", type="date", description="입찰 참여 마감일"),
                AttributeInfo(name="summary", type="string", description="사업 요약"),
                AttributeInfo(name="filename", type="string", description="파일명")
            ],
            search_kwargs={"k": Config.TOP_K}, # "search_type": "mmr", "fetch_k": 20, "lambda_mult": 0.5
            verbose=True  # 쿼리 파싱 과정을 확인하려면 True로 설정
        )        
        # self.retriever = self.vectorstore.as_retriever(
        #     # search_type="similarity",  # search_kwargs={"k": Config.TOP_K}
        #     search_type="mmr", search_kwargs={"k": Config.TOP_K, "fetch_k": 20, "lambda_mult": 0.5}
        # )

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

    def create_chain(self):
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        def format_debug_docs(docs):
            return "\n(-------------------------------------)\n".join(f'[{doc.metadata["filename"]}]\n{doc.page_content}' for doc in docs)

        def get_context(inputs):
            question = inputs["question"]
            docs = self.retriever.get_relevant_documents(question)
            print(format_debug_docs(docs))
            return format_docs(docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 주어진 컨텍스트를 기반으로 사용자의 질문에 답변하는 AI 어시턴트입니다. "
                       "컨텍스트와 대화 이력을 최대한 활용하여 빠짐없이, 상세하게 답변하세요. "
                       "답을 알 수 없는 경우 '제공된 정보만으로는 답변할 수 없습니다.'라고 말하세요. "
                       "추측하지 말고 반드시 한국어로 답변하세요."),
            MessagesPlaceholder("history"),
            ("human", "{question}\n\nContext:\n{context}")
        ])

        chain = (
            RunnablePassthrough.assign(context=get_context)
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.with_config(
            {"callbacks": [self.tracer]}
        )

    async def get_answer(self, query: str) -> str:
        config = RunnableConfig(
            callbacks=[self.tracer, self.rag_handler]
        )
        
        response = await self.chain.ainvoke(
            {"query": query},
            config=config
        )
        
        # Generate visualization
        self.rag_handler.visualize("rag_process.png")
        
        return response

    def ask(self, query: str) -> str:
        response = self.chain.invoke({
            "question": query,
            "history": self.history
        })

        self.history.append(HumanMessage(content=query))
        self.history.append(AIMessage(content=response))
        self.save_history()

        return response