import os
import json
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
from typing import List, Dict

from chain_router import ChainRouter
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from config import Config, LangSmithConfig
from rag_graph import RAGCallbackHandler


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
        router = ChainRouter(llm=self.llm,  retriever=self.retriever, vectorstore=self.vectorstore,
            find_documents_func=self.find_documents, find_contexts_func=self.find_contexts
        )
        self.chain = router.create_router_chain()
        #self.chain = self.create_router_chain() # self.create_chain()
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

    def find_documents(self, question):
        return self.retriever.get_relevant_documents(question)
    
    def find_contexts(self, docs):
        return [(doc.page_content + '\n' + json.dumps(doc.metadata, ensure_ascii=False)) for doc in docs]

    # def create_chain(self):
    #     def format_docs(docs):
    #         return "\n\n".join(self.find_contexts(docs))
    #     def format_debug_docs(docs):
    #         ## return "\n(-------------------------------------)\n".join(docs)
    #         return "\n(-------------------------------------)\n".join(f'[{doc.metadata["filename"]}]\n{doc.page_content}' for doc in docs)

    #     def get_context(inputs):
    #         question = inputs["input"]
    #         docs = self.find_documents(question)
    #         print(format_docs(docs))
    #         return format_docs(docs)

    #     prompt = ChatPromptTemplate.from_messages([
    #         ("system", """당신은 B2G 입찰지원 전문 컨설팅 회사 '입찰메이트'의 최고 수준 AI 어시스턴트입니다.
    #         당신의 주요 임무는 주어진 RFP(제안요청서) 문서 내용을 바탕으로 컨설턴트의 질문에 빠르고 정확하게 답변하는 것입니다.
    #         **규칙:**
    #         1.  **RFP 전문가의 말투 사용**: 답변은 항상 명확하고, 간결하며, 전문가적인 톤을 유지해야 합니다.
    #         2.  **컨텍스트 기반 답변**: 제공된 `Context` 섹션의 내용만을 근거로 답변해야 합니다. 컨텍스트에 없는 내용은 절대로 추측하거나 만들어서는 안 됩니다.
    #         3.  **정보 부족 시 명확한 고지**: 컨텍스트에서 질문에 대한 답을 찾을 수 없다면, "제공된 RFP 문서 내용만으로는 해당 정보를 확인할 수 없습니다."라고 명확하게 답변하세요.
    #         4.  **핵심 정보 요약**: 답변 시에는 예산, 사업 기간, 주요 요구사항 등 핵심적인 정보를 우선적으로, 구조화하여(예: 항목별 리스트) 제공하는 것이 좋습니다.
    #         5.  **한국어 답변**: 모든 답변은 반드시 한국어로 작성해야 합니다."""),
    #         MessagesPlaceholder("history"),
    #         ("human", "{input}\n\nContext:\n{context}")
    #     ])

    #     chain = (
    #         RunnablePassthrough.assign(context=get_context)
    #         | prompt
    #         | self.llm
    #         | StrOutputParser()
    #     )
    #     return chain.with_config(
    #         {"callbacks": [self.tracer]}
    #     )

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