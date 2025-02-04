import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
import uuid, time, os, re
from collections import OrderedDict

# 세션 ID 파일 경로 설정
SESSION_ID_FILE = "session_id.txt"

# 세션 ID 불러오기 또는 생성하기
def get_or_create_session_id():
    if os.path.exists(SESSION_ID_FILE):
        with open(SESSION_ID_FILE, "r") as f:
            session_id = f.read().strip()
            if session_id:
                return session_id
    # 파일이 없거나 내용이 비어있으면 새로 생성
    session_id = str(uuid.uuid4())
    with open(SESSION_ID_FILE, "w") as f:
        f.write(session_id)
    return session_id


# 세션 관리용 In-Memory Store
if "messages" not in st.session_state:
    st.session_state["messages"] = OrderedDict()

# 세션 ID 설정
if "session_id" not in st.session_state:
    st.session_state["session_id"] = get_or_create_session_id()

# Streamlit 페이지 설정
st.set_page_config(page_title="Artwork Chatbot", layout="wide")
st.title('🤖 미술작품 QA 챗봇')


session_id = st.session_state["session_id"]

# 디버깅: 현재 세션 상태 출력
st.write("[DEBUG] Current session_state:")
st.write(st.session_state)

# 이전 대화 내용 표시
st.subheader("대화 기록")
for message_id, message_data in st.session_state["messages"].items():
    if message_data["type"] == "user":
        st.write(f"**사용자 ({message_id}):** {message_data['content']}")
    elif message_data["type"] == "ai":
        st.write(f"**AI ({message_id}):** {message_data['content']}")

@st.cache_resource
# EXAONE 모델 설정
def load_pipeline(model_id):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="cuda",
        trust_remote_code=True,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        top_k=50,
        repetition_penalty=1.05,
    )
    return HuggingFacePipeline(pipeline=pipe)

# 프롬프트 템플릿 생성
def load_prompt_template():
    template = '''
    <|system|>
    You are a friendly chatbot specializing in artworks. 
    Answer questions strictly based on the information provided in the document (context). 
    If the requested information is not found in the document, respond with "The document does not contain this information." 
    Provide detailed and comprehensive answers, always include the artwork number, and ensure all answers are written in Korean. 
    All answers should be formatted using beautiful Markdown syntax to make the response visually appealing and easy to read. 
    Use headings, bullet points, and bold or italic text where appropriate to enrich the response.

    <|context|>
    {context}

    <|user|>
    Question: {question}

    <|assistant|>
    '''

    return ChatPromptTemplate.from_template(template)




# 응답 텍스트에서 답변 추출
def extract_answer(text):
    match = re.search(r"<\|assistant\|>\s*(.*)", text, re.DOTALL)
    if match:
        extracted_text = match.group(1).strip()
        # 불필요한 들여쓰기 제거
        cleaned_text = re.sub(r"^\s{2,}", "", extracted_text, flags=re.MULTILINE)
        return f"### 모델 결과\n\n{cleaned_text}\n"
    else:
        return f"### 모델 결과\n\n{text.strip()}\n"




# 대화 기록 생성
def generate_chat_history():
    chat_history = []
    for message_id, message_data in st.session_state["messages"].items():
        if message_data["type"] == "user":
            chat_history.append(f"User: {message_data['content']}")
        elif message_data["type"] == "ai":
            chat_history.append(f"AI: {message_data['content']}")
    return "\n".join(chat_history)

# FAISS 데이터베이스 설정
faiss_path = "./faiss_artworks_0114"
embedding_model = SentenceTransformer("nlpai-lab/KURE-v1")
with st.spinner("Loading FAISS database..."):
    faiss_db = FAISS.load_local(
        folder_path=faiss_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    st.success("FAISS database loaded successfully!")

faiss_db.embedding_function = lambda text: embedding_model.encode(text)

# LLM 및 프롬프트 초기화
llm = load_pipeline("LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
prompt = load_prompt_template()

retriever = faiss_db.as_retriever(
    search_kwargs={
        "k": 5,
        "fetch_k": 15,
        "mmr": True,
        "mmr_beta": 0.8
    }
)

# 사용자 입력 처리
with st.form("Question"):
    user_input = st.text_area("질문 내용:", "질문을 입력하세요.")
    submitted = st.form_submit_button("보내기")

if submitted:
    try:
        # 메시지 ID 생성
        user_message_id = str(uuid.uuid4())
        ai_message_id = str(uuid.uuid4())

        # 사용자 메시지 저장
        st.session_state["messages"][user_message_id] = {"type": "user", "content": user_input}

        # 문맥 검색 (캐싱 적용)
        start_time = time.time()
        context_documents = retriever.get_relevant_documents(user_input)
        context = "\n\n".join([doc.page_content for doc in context_documents])
        chat_history = generate_chat_history()

        # 프롬프트 생성
        formatted_prompt = prompt.format(
            chat_history=chat_history,
            context=context,
            question=user_input
        )

        # 모델 호출 
        response = llm.invoke(formatted_prompt)
        response_text = extract_answer(response["generated_text"] if isinstance(response, dict) else response)
        print(type(response_text))
        # AI 메시지 저장
        st.session_state["messages"][ai_message_id] = {"type": "ai", "content": response_text}

        # 현재 질문과 답변 출력
        st.markdown("---")
        st.markdown("### 현재 대화")
        st.markdown(f"**질문:** {user_input}")
        st.markdown(f"**답변:**\n\n")
        st.markdown(response_text, unsafe_allow_html=False)
        print(response_text)
        # 응답 시간 출력
        end_time = time.time()
        st.write(f"응답 시간: {end_time - start_time:.2f}초")

    except Exception as e:
        # 오류 처리
        st.error(f"오류 발생: {str(e)}")
