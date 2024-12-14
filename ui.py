import streamlit as st

def setup_ui():
    st.set_page_config(page_title="Page Title")
    st.title('Title : Streamlit Test')

def get_user_input():
    with st.form('Question'):
        text = st.text_area('질문 내용:', '질문 기본값')
        submitted = st.form_submit_button('보내기')
        return text, submitted

def display_response(response):
    st.info(response)
