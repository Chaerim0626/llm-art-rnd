import streamlit as st
from model import load_pipeline
from database import load_database
from chain import load_prompt_template, process_input
from memory import create_summary_memory
import uuid
from langchain.schema import HumanMessage, AIMessage  # 메시지 클래스 임포트

# Streamlit UI 설정
st.set_page_config(page_title="Artwork Chatbot")
st.title('🤖 미술작품 QA 챗봇')

# 세션 ID 관리
session_id = st.session_state.get('session_id', None)
if not session_id:
    session_id = str(uuid.uuid4())
    st.session_state['session_id'] = session_id

# EXAONE 모델 로드
llm = load_pipeline('LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct')

# Memory 생성
memory = create_summary_memory(session_id, llm)

# 데이터베이스 초기화
persist_directory = "../chroma_1205"
collection_name = "chroma_art"
retriever = load_database(persist_directory, collection_name)

# 프롬프트 템플릿 로드
prompt = load_prompt_template('prompt.yaml')

# 이전 대화 내용 표시
st.subheader("대화 기록")
if memory.chat_memory.messages:
    for message in memory.chat_memory.messages:
        if isinstance(message, HumanMessage):  # 사용자 메시지
            st.write(f"**사용자:** {message.content}")
        elif isinstance(message, AIMessage):  # AI 모델 메시지
            st.write(f"**이전 답변:** {message.content}")
else:
    st.write("아직 대화 기록이 없습니다.")

# 사용자 입력 처리
with st.form('Question'):
    user_input = st.text_area('질문 내용:', '질문을 입력하세요.')
    submitted = st.form_submit_button('보내기')

    if submitted:
        # 모델 응답 생성
        response = process_input(llm, retriever, prompt, user_input)

        # 대화 기록 업데이트
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(response)

        # 현재 질문과 답변 출력
        st.markdown("---")
        st.markdown("### 현재 대화")
        st.write(f"**질문:** {user_input}")
        st.markdown(f"**답변:** {response}")
