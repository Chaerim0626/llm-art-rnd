import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma as ChromaStore
from langchain.document_loaders import Docx2txtLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers.string import StrOutputParser
from langchain import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

# Streamlit 설정
st.set_page_config(page_title="Page Title ")
st.title('Title : Streamlit Test')

# 모델과 토크나이저 초기화
model_name = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map="cuda",
    trust_remote_code=True
)

# 파이프라인 생성
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,  # 생성할 최대 토큰 수 증가
    do_sample=True,        # 샘플링 활성화
    temperature=0.8,      # 다양성 증가
    top_k=50,             # 상위 k개 토큰 중에서 샘플링
    repetition_penalty=1.03
)

llm = HuggingFacePipeline(pipeline=pipe)

# 문서 로드
# 기존 Chroma 데이터베이스 사용
persist_directory = './chroma_huggingface1029'  # 기존 Chroma DB 경로
embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large')
database = ChromaStore(persist_directory=persist_directory, embedding_function=embeddings)
retriever = database.as_retriever(search_kwargs={"k": 2})


# Prompt 템플릿 정의
template = '''

'''
prompt = ChatPromptTemplate.from_template(template)


def extract_answer(text):
    """ 정규 표현식을 사용하여 'Answer:' 이후의 모든 텍스트 추출 """
    match = re.search(r"Answer:\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()  # 불필요한 공백 제거
    return "답변을 찾을 수 없습니다."


# 체인 구성
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def response(input_text):
    """ 입력받은 input text에 대해 모델의 응답을 출력 """
    context = retriever.get_relevant_documents(input_text)
    context_text = " ".join(doc.page_content for doc in context) if context else "관련된 정보가 없습니다."
    
    input_text_with_context = prompt.format(context=context_text, question=input_text)
    model_response = pipe(input_text_with_context)
    
    # 응답에서 'Answer:' 추출
    return extract_answer(model_response[0]['generated_text'])

with st.form('Question'):
    text = st.text_area('질문 내용:', '질문 기본값')  # 첫 페이지가 실행될 때 보여줄 질문
    submitted = st.form_submit_button('보내기')  # 버튼 이름
    
    if submitted:
        # 모델의 응답 출력
        model_response = response(text)
        st.info(model_response)
