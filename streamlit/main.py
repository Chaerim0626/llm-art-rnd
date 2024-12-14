import streamlit as st
from model import load_pipeline
from database import load_database
from chain import load_prompt_template, create_chain, process_input

# Streamlit UI 설정
st.set_page_config(page_title="Artwork Chatbot")
st.title('🤖 미술작품 QA 챗봇')

# 모델 및 데이터베이스 초기화
llm = load_pipeline('LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct')
# 데이터베이스 경로 및 컬렉션 이름
persist_directory = "../chroma_1205"
collection_name = "chroma_art"
retriever = load_database(persist_directory, collection_name)


# 프롬프트 템플릿 로드
prompt = load_prompt_template('prompt.yaml')

# 체인 생성
chain = create_chain(retriever, prompt, llm)

# 사용자 입력 처리
with st.form('Question'):
    user_input = st.text_area('질문 내용:', '질문을 입력하세요.')
    submitted = st.form_submit_button('보내기')

    if submitted:
        # 체인을 통해 응답 생성
        response = process_input(llm, retriever, prompt, user_input)
        st.markdown(response)
