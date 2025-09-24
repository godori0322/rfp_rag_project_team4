import streamlit as st
from chatbot import Chatbot  # src/main.py -> chatbot.py
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. 페이지 설정 ---
st.set_page_config(
    page_title="입찰메이트 RFP 분석",
    page_icon="🤝",
    layout="wide"
)

# --- 2. 세션 상태 초기화 ---
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. 사이드바: 사용자 인증 및 세션 시작 ---
with st.sidebar:
    st.image("https://i.imgur.com/g055F5d.png", width=250)
    st.header("입찰메이트 🤝")
    st.markdown("RFP 문서 기반 질의응답 시스템")
    st.divider()

    user_id_input = st.text_input("사용자 ID를 입력하세요.", key="user_id_input")

    if st.button("세션 시작/전환", use_container_width=True):
        if user_id_input:
            st.session_state.user_id = user_id_input
            # Chatbot 인스턴스를 생성하면 __init__에서 자동으로 히스토리를 로드합니다.
            st.session_state.chatbot = Chatbot(st.session_state.user_id)
            
            # UI에 표시할 메시지 목록을 생성합니다.
            st.session_state.messages = []
            for msg in st.session_state.chatbot.history:
                if isinstance(msg, HumanMessage):
                    st.session_state.messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    st.session_state.messages.append({"role": "assistant", "content": msg.content})
            
            st.success(f"'{st.session_state.user_id}'님, 안녕하세요! 이전 대화 기록을 불러왔습니다.")
            # st.rerun()을 호출하여 메인 화면을 즉시 새로고침합니다.
            st.rerun() 
        else:
            st.error("사용자 ID를 입력해야 합니다.")
    st.divider()
    st.caption("© 2025 입찰메이트 Engineering Team.")

# --- 4. 메인 화면: 채팅 인터페이스 ---
st.title("📑 RFP 문서 분석 및 질의응답")

if not st.session_state.chatbot:
    st.info("사이드바에서 사용자 ID를 입력하고 세션을 시작해주세요.")
else:
    st.success(f"현재 세션: **{st.session_state.user_id}**")

    # 이전 대화 기록 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("RFP 문서에 대해 질문해보세요."):
        # 사용자가 입력한 내용을 UI에 먼저 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Chatbot의 ask 메서드를 호출하여 답변을 받아옴
        with st.spinner("AI가 답변을 생성하는 중입니다..."):
            response = st.session_state.chatbot.ask(prompt)

        # AI의 답변을 UI에 표시
        # ask 메서드가 chatbot.history를 자동으로 업데이트하므로 UI용 messages 목록에만 추가
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)