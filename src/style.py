# style.py

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

    /* 다른 CSS 스타일들을 추가 */

    </style>
    """, unsafe_allow_html=True)