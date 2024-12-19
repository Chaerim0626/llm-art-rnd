import streamlit as st

st.set_page_config(page_title="Artwork Chatbot", layout="wide")
from model import load_pipeline
from database import load_database
from chain import load_prompt_template, process_input
import uuid
from langchain.schema import HumanMessage, AIMessage
from model import llm
from config import memory
from summary import extract_summary
from langchain.memory import VectorStoreRetrieverMemory
from langchain_huggingface import HuggingFaceEmbeddings  # 변경된 임포트
from langchain_chroma import Chroma  # 변경된 임포트


st.title("🖼️ Artwork QA Chatbot")

# 세션 ID 관리
session_id = st.session_state.get('session_id', None)
if not session_id:
    session_id = str(uuid.uuid4())
    st.session_state['session_id'] = session_id

# 데이터베이스 초기화
persist_directory = "../chroma_1218"
collection_name = "chroma_art"
retriever = load_database(persist_directory, collection_name)

embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large')

# 대화용 DB 초기화
chat_persist_directory = "../chroma_chat_db"  # 대화 DB 저장 경로
chat_vectorstore = Chroma(persist_directory=chat_persist_directory, embedding_function=embeddings)
chat_retriever = chat_vectorstore.as_retriever()

# VectorStoreRetrieverMemory 설정
chat_memory = VectorStoreRetrieverMemory(
    retriever=chat_retriever,
    memory_key="history",
    input_key="question"
)

def save_to_chat_memory(question, answer):
    chat_memory.save_context({"question": question}, {"output": answer})
    
# 프롬프트 템플릿 로드
prompt = load_prompt_template('prompt.yaml')

# 이전 대화 내용 표시



# 사용자 입력 처리 폼 (조건문 밖에 위치)
with st.form('Question_form'):  # 폼의 key를 명시적으로 설정
    user_input = st.text_area('질문 내용:', '질문을 입력하세요.')
    submitted = st.form_submit_button('보내기')

# 폼 제출 처리
if submitted:  # 폼이 제출되었을 때만 실행
    # 모델 응답 생성
    response = process_input(llm, retriever,chat_retriever, prompt, user_input, chat_memory)
    if not response:
        response = "죄송합니다, 답변을 생성할 수 없습니다."

    # VectorStore 메모리 업데이트 (대화 저장)
    chat_memory.save_context(
        {"question": user_input},  # 사용자 입력
        {"output": response}       # 모델 응답
    )

    # 현재 질문과 답변 출력
    st.markdown("---")
    st.markdown("### 현재 대화")
    st.write(f"**질문:** {user_input}")
    st.markdown(f"**답변:** {response}")

#대화 요약
if st.button("대화 요약 보기"):
    try:
        memory_variables = chat_memory.load_memory_variables({"question": ""})  # 빈 question 명시
        history = memory_variables.get("history", "")

        # 요약 표시
        st.markdown("### 대화 요약:")
        if history:
            st.write(memory_variables)
        else:
            st.write("요약된 대화가 없습니다.")
    except Exception as e:
        st.error(f"오류 발생: {e}")