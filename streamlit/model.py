import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


# 캐싱된 모델 로드
@st.cache_resource
def load_pipeline(model_name, batch_size=4):  # 배치 크기를 매개변수로 추가
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="cuda",
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
        repetition_penalty=1.03,
        batch_size=batch_size  # 배치 크기 설정
    )
    return HuggingFacePipeline(pipeline=pipe)


# EXAONE 모델 로드
llm = load_pipeline('LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct')
