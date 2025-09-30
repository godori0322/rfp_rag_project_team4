import streamlit as st
from chatbot import Chatbot
from langchain_core.messages import HumanMessage, AIMessage
from style import apply_custom_css

# CSS 설정 적용
apply_custom_css()

# --- 페이지 설정 ---
st.set_page_config(
    page_title="입찰메이트 RFP 분석",
    page_icon="🤝",
    layout="wide"
)

# --- 세션 상태 초기화 ---
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 사이드바 ---
with st.sidebar:
    st.image("data/image/rogo.png", width=250)
    st.header("입찰메이트 🤝")
    st.markdown("RFP 문서 기반 질의응답 시스템")
    st.divider()
    with st.form(key="session_form"):
        user_id_input = st.text_input(
            "사용자 ID를 입력하세요.",
            key="user_id_input",
            placeholder="아이디 입력 후 Enter"
        )
        submit_button = st.form_submit_button(
            label="세션 시작/전환",
            use_container_width=True
        )
    if submit_button:
        if user_id_input:
            st.session_state.user_id = user_id_input
            st.session_state.chatbot = Chatbot(st.session_state.user_id)
            st.session_state.messages = []
            for msg in st.session_state.chatbot.history:
                if isinstance(msg, HumanMessage):
                    st.session_state.messages.append({"role": "user", "answer": msg.content})
                elif isinstance(msg, AIMessage):
                    st.session_state.messages.append({
                        "role": "assistant", "answer": msg.content,
                        "inference_time": None, "context_docs": None
                    })
            st.success(f"'{st.session_state.user_id}'님, 안녕하세요! 이전 대화 기록을 불러왔습니다.")
        else:
            st.error("사용자 ID를 입력해야 합니다.")
    st.divider()
    st.caption("© 2025 입찰메이트 Engineering Team.")

# --- 메인 화면: 채팅 인터페이스 ---
st.title("📑 RFP 문서 분석 및 질의응답")

# --- 아이콘 경로 설정 ---
USER_AVATAR = "👤"
BOT_AVATAR = "data/image/rogo.png"

if not st.session_state.chatbot:
    st.info("사이드바에서 사용자 ID를 입력하고 세션을 시작해주세요.")
else:
    st.success(f"현재 세션: **{st.session_state.user_id}**")
    
    # --- 초기 환영 메시지 및 예시 질문 추가 ---
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown("안녕하세요! 입찰메이트 AI 컨설턴트 비디(BIDI)입니다. 🤝")
            st.markdown("RFP 문서에 대해 궁금한 점을 질문해주세요. 아래와 같은 질문들을 할 수 있습니다.")
            with st.container(border=True):
                st.markdown("- **최근 1달간** 공개된 **AI 관련 사업** 목록 보여줘")
                st.markdown("- **'한국지능정보사회진흥원'** 에서 발주한 사업 찾아줘")
                st.markdown("- 사업 예산이 **10억 이상**인 공고만 검색해줘")

    # 이전 대화 기록 출력
    for message in st.session_state.messages:
        role = message["role"]
        avatar = BOT_AVATAR if role == "assistant" else USER_AVATAR
        with st.chat_message(role, avatar=avatar):
            text = message.get("answer") or message.get("content") or ""
            st.markdown(text)
            if role == "assistant":
                if message.get("inference_time") is not None:
                    st.caption(f"⏱ 추론 시간: {message['inference_time']:.2f}초")

    # 사용자 입력 처리
    if prompt := st.chat_input("RFP 문서에 대해 질문해보세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        with st.spinner("AI가 답변을 생성하는 중입니다..."):
            response = st.session_state.chatbot.ask(prompt)

        new_message = {
            "role": "assistant",
            "answer": response["answer"],
            "inference_time": response.get("inference_time"),
            "context_docs": response.get("context_docs")
        }
        st.session_state.messages.append(new_message)

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown(new_message["answer"])
            st.caption(f"⏱ 추론 시간: {new_message['inference_time']:.2f}초")