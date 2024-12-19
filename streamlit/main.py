
import streamlit as st
st.set_page_config(page_title="Artwork Chatbot", layout="wide")
import uuid
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable
from langchain.schema import HumanMessage, AIMessage

from model import llm  # EXAONE 모델 로드
from database import load_database  # 사용자 정의 database 로드 함수
import re
# Streamlit 설정


# Database 초기화
persist_directory = "../chroma_1205"
collection_name = "chroma_art"
database = load_database(persist_directory, collection_name)

# Retriever 정의
retriever = database.as_retriever(search_kwargs={"k": 5})  # 상위 5개 결과 검색

# 세션 관리용 In-Memory Store
if "store" not in st.session_state:
    st.session_state.store = {}
store = st.session_state.store
# Streamlit UI
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

session_id = st.session_state["session_id"]

# 세션 기록 가져오기 함수
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    global store
    if session_id not in store.keys():
        store[session_id] = InMemoryChatMessageHistory()  # InMemoryChatMessageHistory 객체
    return store[session_id]


# 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_template(
    template = """<|system|>
You are a friendly chatbot specializing in artworks. 
Answer questions strictly based on the information provided in the document (context). 
Provide detailed and comprehensive answers, always include the artwork number, and ensure all answers are written in Korean.

You must format all responses using Markdown to make them visually appealing and organized.
Use the following Markdown guidelines to improve readability and structure:
- Use headings (`#`, `##`, `###`) to structure the information.
- Use bullet points or numbered lists to break down details clearly.
- Highlight key terms with **bold** or _italic_ formatting.
- Add horizontal rules (`---`) to separate sections for better clarity.
- Include block quotes (`>`) for emphasized information or quotes.
- Utilize tables for structured data when appropriate (e.g., artwork details like 제작 연도, 크기, 재료, etc.).
- Add line breaks for a clean and professional layout.

Always consider the conversation history (`chat_history`) when answering the user's question. 
If the user's current question relates to or builds upon a previous discussion, incorporate relevant details from `chat_history` into your answer.

<|chat_history|>
{chat_history}

<|context|>
{context}

<|user|>
Question: {question}

<|assistant|>
"""
)



# MarkdownOutputParser 정의
class MarkdownOutputParser:
    """Enhanced Markdown parser with additional formatting options."""
    def __call__(self, llm_output):
        # <|assistant|> 이후의 텍스트만 추출
        match = re.search(r"<\|assistant\|>\s*(.*)", llm_output, re.DOTALL)
        if match:
            extracted_text = match.group(1).strip()
            # 마크다운 형식으로 반환
            return f"\n{extracted_text}\n"
        else:
            # <|assistant|> 태그가 없는 경우 원래 출력 반환
            return f"\n{llm_output.strip()}\n"

# Runnable 정의
class ProcessInputRunnable(Runnable):
    def __init__(self, parser):
        self.parser = parser

    def invoke(self, input_data: dict, config: dict) -> str:
        input_text = input_data.get("input", "No input provided")
        session_id = config.get("configurable", {}).get("session_id", "Unknown session")
        
        # 세션 기록 가져오기
        history = get_session_history(session_id)

        # 사용자 메시지 추가
        user_message = HumanMessage(content=input_text)
        history.add_user_message(user_message)

        # 대화 기록을 텍스트로 변환
        history_text = "\n".join(
            f"User: {message.content}" if isinstance(message, HumanMessage) else f"AI: {message.content}"
            for message in history.messages
        )
        print("store:", store)
        print("history_text:", history_text)

        # 검색 수행
        search_results = retriever.get_relevant_documents(input_text)
        context = "\n".join([result.page_content for result in search_results])

        # 프롬프트 생성
        if context:
            # EXAONE 모델 호출

            formatted_prompt = prompt.format(
                chat_history=history_text,
                context=context,
                question=input_text
            )
                     
            llm_response = llm(formatted_prompt)
            # 디버깅: LLM 응답 확인
            print(f"[DEBUG] LLM 응답의 타입: {type(llm_response)}")

            # 응답 처리
            response_text = llm_response if isinstance(llm_response, str) else "No response generated"
            response_text = self.parser(response_text)


            # AI 메시지 추가
            ai_message = AIMessage(content=response_text)
            history = get_session_history(session_id)
            history.add_message(ai_message)
            print("[DEBUG] Updated chat history:", history.messages)


        print("[DEBUG] chat_history:", history)

        return response_text


# ProcessInputRunnable 인스턴스 생성
markdown_parser = MarkdownOutputParser()
process_input_runnable = ProcessInputRunnable(parser=markdown_parser)

# RunnableWithMessageHistory 정의
with_message_history = RunnableWithMessageHistory(
    runnable=process_input_runnable,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


st.title("🖼️ Artwork QA Chatbot")

# 사용자 입력 처리
with st.form("input_form"):
    user_input = st.text_area("질문 입력", "여기에 질문을 입력하세요.")
    submitted = st.form_submit_button("보내기")

if submitted:
    input_data = {"input": user_input}
    config = {"configurable": {"session_id": session_id}}
    response = with_message_history.invoke(input_data, config)
    st.markdown(response)  # Markdown 형식으로 출력