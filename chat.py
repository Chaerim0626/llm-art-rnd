import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
import re
import uuid

# Streamlit UI 설정
@st.cache_resource

# EXAONE 모델 설정
@st.cache_resource
def load_pipeline(model_id):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="cuda",  # CUDA에서 자동 배치
        trust_remote_code=True
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        top_k=50,
        repetition_penalty=1.05
    )
    return HuggingFacePipeline(pipeline=pipe)

# 프롬프트 템플릿 생성
def load_prompt_template():
    template = '''
<|system|>
  You are a friendly chatbot specializing in artworks. 
  Answer questions strictly based on the information provided in the document (context). 
  If the requested information is not found in the document, respond with "The document does not contain this information." 
  Provide detailed and comprehensive answers, always include the artwork ID, and ensure all answers are written in polite and grammatically correct Korean.
  Your tone should always be friendly, formal, and respectful.
  If the question is unclear or incomplete, politely ask for clarification in Korean.
  If possible, provide related details from the context that may enrich the user's understanding of the artwork.

<|context|>
{context}

<|user|>
Question: {question}

<|assistant|>
'''
    return ChatPromptTemplate.from_template(template)

def extract_answer(text):
    """
    응답 텍스트에서 'Answer:' 또는 특정 패턴 이후 텍스트를 추출.
    - text: 모델에서 생성된 텍스트
    """
    match = re.search(r"<\|assistant\|>\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "답변을 찾을 수 없습니다."

def process_input(llm, retriever, prompt, user_input):
    """
    입력된 질문과 문맥(Context)을 결합하여 모델에 전달.
    - llm: 모델 파이프라인 객체
    - retriever: 데이터베이스 검색 객체
    - prompt: ChatPromptTemplate 객체
    - user_input: 사용자 입력 질문
    """
    # 문맥(Context) 생성
    context = retriever.invoke(user_input)
    context_text = " ".join(doc.page_content for doc in context) if context else "관련된 정보가 없습니다."
    
    # 프롬프트 생성
    input_text_with_context = prompt.format(context=context_text, question=user_input)
    
    # 모델 응답 생성
    response = llm.pipeline(input_text_with_context)
    
    # 응답에서 <|assistant|> 이후 텍스트 추출
    return extract_answer(response[0])


# MarkdownOutputParser 정의
class MarkdownOutputParser:
    def __call__(self, llm_output):
        match = re.search(r"<\|assistant\|>\s*(.*)", llm_output, re.DOTALL)
        if match:
            extracted_text = match.group(1).strip()
            return f"### 모델 결과\n\n{extracted_text}\n\n"
        else:
            return f"### 모델 결과\n\n{llm_output.strip()}\n\n"

# 세션 설정
st.set_page_config(page_title="Artwork Chatbot")
st.title('🤖 미술작품 QA 챗봇')

# 세션 ID 관리
session_id = st.session_state.get('session_id', None)
if not session_id:
    session_id = str(uuid.uuid4())
    st.session_state['session_id'] = session_id

# FAISS 데이터베이스 설정
faiss_path = "./faiss_artworks"
embedding_model = SentenceTransformer("nlpai-lab/KURE-v1")
with st.spinner("Loading FAISS database..."):
    faiss_db = FAISS.load_local(
        folder_path=faiss_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    st.success("FAISS database loaded successfully!")

# embedding_function 디버깅
def debug_embedding_function(text):
    print(f"Debug: embedding_function input type: {type(text)}")
    print(f"Debug: embedding_function input: {text}")
    return embedding_model.encode(text)

faiss_db.embedding_function = debug_embedding_function

# LLM 및 프롬프트 초기화
llm = load_pipeline("LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
prompt = load_prompt_template()

retriever = faiss_db.as_retriever(
    search_kwargs={
        "k": 5,                # 검색 결과 개수
        "fetch_k": 20,         # 더 많은 결과 가져오기
        "mmr": True,           # MMR 활성화
        "mmr_beta": 0.5        # 다양성과 관련성 간 균형
    }
)

# 체인 구성
chain = (
    RunnableMap({
        "context": lambda query: retriever.get_relevant_documents(query["question"]),  # Retriever 호출
        "question": RunnablePassthrough()
    })
    | (lambda x: {
        "context": "\n\n".join([doc.page_content for doc in x["context"]]),  # 문서 내용을 문자열로 변환
        "question": x["question"]
    })
    | prompt  # Prompt Template에 전달
    | llm     # LLM 호출
    | MarkdownOutputParser()  # Markdown 포맷으로 출력
)


# 이전 대화 내용 표시
st.subheader("대화 기록")
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

for message in st.session_state['messages']:
    if isinstance(message, HumanMessage):
        st.write(f"**사용자:** {message.content}")
    elif isinstance(message, AIMessage):
        st.write(f"**이전 답변:** {message.content}")

# 사용자 입력 처리
with st.form('Question'):
    user_input = st.text_area('질문 내용:', '질문을 입력하세요.')
    submitted = st.form_submit_button('보내기')

    if submitted:
        with st.spinner("Processing your query..."):
            try:
                # 체인 실행
                response = chain.invoke({"question": user_input})

                # 대화 기록 업데이트
                st.session_state['messages'].append(HumanMessage(content=user_input))
                st.session_state['messages'].append(AIMessage(content=response))

                # 현재 질문과 답변 출력
                st.markdown("---")
                st.markdown("### 현재 대화")
                st.write(f"**질문:** {user_input}")
                st.markdown(f"**답변:**\n{response}")
            except Exception as e:
                st.error(f"오류 발생: {e}")
