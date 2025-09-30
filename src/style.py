import streamlit as st

def apply_custom_css():
    """
    Streamlit 앱에 적용할 전체 커스텀 CSS를 관리하는 함수
    """
    st.markdown("""
    <style>
    /* 폼 테두리 제거 */
    [data-testid="stForm"] {
        border: none;
    }

    /* --- 채팅 메시지 버블 스타일 --- */

    /* 메시지 버블 자체에 대한 공통 스타일 */
    div[data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] {
        border-radius: 15px;
        padding: 1.1rem;
    }

    /* AI 메시지 버블 스타일 */
    div[data-testid="stChatMessage"]:not(:has(div[style*="flex-direction: row-reverse;"])) div[data-testid="stChatMessageContent"]  {
        background-color: #FFFFFF;
        color: #333333;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); /* 그림자 효과 */
        border: 1px solid rgba(0,0,0,0.06);
    }

    /* 사용자 메시지 버블 스타일 */
    div[data-testid="stChatMessage"]:has(div[style*="flex-direction: row-reverse;"]) div[data-testid="stChatMessageContent"] {
        background-color: #e1f0ff;
        color: #333333;
        box-shadow: none; /* 그림자 제거 */
        border: 1px solid #C4DEFF;
    }

    </style>
    """, unsafe_allow_html=True)