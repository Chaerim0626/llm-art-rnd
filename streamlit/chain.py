import re
import yaml
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain.chains import ConversationChain

class MarkdownOutputParser:
    """Enhanced Markdown parser with additional formatting options."""

    def __call__(self, llm_output):
        # <assistant> 이후의 텍스트만 추출
        match = re.search(r"<\|assistant\|>\s*(.*)", llm_output, re.DOTALL)
        if match:
            extracted_text = match.group(1).strip()
            # 마크다운 코드 블록으로 출력 포맷
            return f"\n{extracted_text}\n"
        else:
            # <assistant> 태그가 없는 경우 원래 출력 반환
            return f"\n{llm_output.strip()}\n"



def load_prompt_template(file_path):
    """
    YAML 파일에서 프롬프트 템플릿을 로드.
    - file_path: YAML 파일 경로
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return ChatPromptTemplate.from_template(data['template'])


def create_chain(retriever, chat_retriever, prompt, llm, memory, input):
    """
    컨텍스트와 대화 이력을 결합해 모델 체인 구성.
    - retriever: 작품 관련 문서 검색
    - chat_retriever: 대화 이력 검색
    """
    def load_context_and_history(input):
        # 작품 문서 검색
        docs = retriever.invoke(input)
        context = "\n".join([doc.page_content for doc in docs])
        
        # 대화 이력 검색
        chat_docs = chat_retriever.invoke(input)
        history = "\n".join([doc.page_content for doc in chat_docs])

        return {"context": context, "history": history}

    return (
        RunnableMap({
            "context": lambda input: load_context_and_history(input)["context"],
            "history": lambda input: load_context_and_history(input)["history"],
            "question": lambda input: input  # question을 명시적으로 입력받아 전달
        })
        | prompt  # 프롬프트에 context, history, question 전달
        | llm  # LLM 실행
        | MarkdownOutputParser()  # 출력 결과를 마크다운으로 포맷팅
    )

def extract_answer(text):
    """
    응답 텍스트에서 'Answer:' 또는 특정 패턴 이후 텍스트를 추출.
    - text: 모델에서 생성된 텍스트
    """
    match = re.search(r"<\|assistant\|>\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "답변을 찾을 수 없습니다."


def process_input(llm, retriever,chat_retriever, prompt, user_input, memory):
    """
    체인을 사용해 입력된 질문과 문맥(Context)을 결합하고 모델을 실행.
    - llm: 모델 파이프라인 객체
    - retriever: 데이터베이스 검색 객체
    - prompt: ChatPromptTemplate 객체
    - user_input: 사용자 입력 질문
    """
    # 체인 생성
    chain = create_chain(retriever,chat_retriever, prompt, llm, memory, user_input)
    
    # 체인 실행: 입력된 질문을 기반으로 결과 반환
    result = chain.invoke(user_input)
    
    # 결과를 반환
    if result:
        return result 
    else:
        return "결과를 생성할 수 없습니다."

