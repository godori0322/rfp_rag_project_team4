import os
import json
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser

from config import Config

class Chatbot:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.history_file = os.path.join(Config.HISTORY_PATH, f"{self.user_id}_history.json")

        load_dotenv(find_dotenv())
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        self.initialize_components()
        self.history = self.load_history()
        self.chain = self.create_chain()

    def initialize_components(self):
        self.embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        self.vectorstore = FAISS.load_local(Config.VECTOR_DB_PATH, embeddings=self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": Config.TOP_K})
        self.llm = ChatOpenAI(model_name=Config.LLM_MODEL, temperature=Config.TEMPERATURE)

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

        def get_context(inputs):
            question = inputs["question"]
            docs = self.retriever.get_relevant_documents(question)
            return format_docs(docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 주어진 컨텍스트를 기반으로 사용자의 질문에 답변하는 AI 어시턴트입니다. 컨텍스트와 대화 이력의 내용을 최대한 활용하여 빠짐없이, 상세하게 답변해주세요. '제공된 정보만으로는 답변을 찾을 수 없습니다.'라고 솔직하게 말하세요. 추측해서 답변하지 마세요. 반드시 한국어로 답변하세요."),
            MessagesPlaceholder("history"),
            ("human", "Question: {question}\n\nContext: {context}")
        ])

        chain = (
            {
                "context": get_context,
                "question": lambda x: x["question"],
                "history": lambda x: x["history"]
            }
            | prompt | self.llm | StrOutputParser()
        )

        return chain

    def ask(self, query: str) -> str:
        response = self.chain.invoke({
            "question": query,
            "history": self.history
        })

        self.history.append(HumanMessage(content=query))
        self.history.append(AIMessage(content=response))
        self.save_history()

        return response