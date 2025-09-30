import streamlit as st
from chatbot import Chatbot  # src/main.py -> chatbot.py
from langchain_core.messages import HumanMessage, AIMessage
from style import apply_custom_css  

# css 설정 관련
apply_custom_css()  # 2. 가져온 함수를 실행하여 스타일을 적용합니다.

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
    st.image("data/image/rogo.png", width=250)
    st.header("입찰메이트 🤝")
    st.markdown("RFP 문서 기반 질의응답 시스템")
    st.divider()

    # st.form을 생성하여 입력 필드와 버튼을 묶습니다.
    with st.form(key="session_form"):
        user_id_input = st.text_input(
            "사용자 ID를 입력하세요.", 
            key="user_id_input",
            placeholder="아이디 입력 후 Enter" # 사용자 편의를 위한 안내 문구
        )
        
        # st.button 대신 st.form_submit_button을 사용합니다.
        submit_button = st.form_submit_button(
            label="세션 시작/전환", 
            use_container_width=True
        )

    # 폼이 제출되었을 때 (버튼 클릭 또는 Enter) 아래 로직을 실행합니다.
    if submit_button:
        if user_id_input:
            st.session_state.user_id = user_id_input
            st.session_state.chatbot = Chatbot(st.session_state.user_id)
            
            st.session_state.messages = []
            for msg in st.session_state.chatbot.history:
                if isinstance(msg, HumanMessage):
                    st.session_state.messages.append({
                        "role": "user",
                        "answer": msg.content
                    })
                elif isinstance(msg, AIMessage):
                    st.session_state.messages.append({
                        "role": "assistant",
                        "answer": msg.content,
                        "inference_time": None,
                        "context_docs": None
                    })
            
            st.success(f"'{st.session_state.user_id}'님, 안녕하세요! 이전 대화 기록을 불러왔습니다.")
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
        role = message["role"]
        with st.chat_message(role):
            # user는 'content', assistant는 'answer' 사용
            text = message.get("answer") or message.get("content") or ""
            st.markdown(text)
        
            if role == "assistant":
                # 메타 정보 출력
                if message.get("inference_time") is not None:
                    st.caption(f"⏱ 추론 시간: {message['inference_time']:.2f}초")
                if message.get("context_docs"):
                    with st.expander("🔍 참조 문서 보기"):
                        for i, doc in enumerate(message["context_docs"], start=1):
                            st.markdown(f"**[{i}]** {doc.page_content[:300]}...")
                            st.caption(f"출처: {doc.metadata.get('filename', 'N/A')}")

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
        st.session_state.messages.append({
            "role": "assistant",
            "answer": response["answer"],          # 실제 텍스트
            "inference_time": response.get("inference_time"),
            "context_docs": response.get("context_docs")
        })
        # assistant 메시지 UI에 표시
        with st.chat_message("assistant"):
            st.markdown(response["answer"])
            st.caption(f"⏱ 추론 시간: {response['inference_time']:.2f}초")
            if response.get("context_docs"):
                with st.expander("🔍 참조 문서 보기"):
                    for i, doc in enumerate(response["context_docs"], start=1):
                        st.markdown(f"**[{i}]** {doc.page_content[:300]}...")
                        st.caption(f"출처: {doc.metadata.get('filename', 'N/A')}")