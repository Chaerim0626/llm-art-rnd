import re
import yaml
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough


class MarkdownOutputParser:
    """Enhanced Markdown parser with additional formatting options."""

    def __call__(self, llm_output):
        """
        모델 출력에서 <|assistant|> 이후의 텍스트를 추출하여 마크다운 형식으로 반환.
        """
        match = re.search(r"<\|assistant\|>\s*(.*)", llm_output, re.DOTALL)
        if match:
            extracted_text = match.group(1).strip()
            return f"### 모델 결과\n\n```markdown\n{extracted_text}\n```\n"
        else:
            return f"### 모델 결과\n\n```markdown\n{llm_output.strip()}\n```\n"


def load_prompt_template(file_path):
    """
    YAML 파일에서 프롬프트 템플릿을 로드.
    - file_path: YAML 파일 경로
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return ChatPromptTemplate.from_template(data['template'])


def create_chain(retriever, prompt, llm):
    """
    체인을 구성.
    - retriever: 데이터베이스 검색기
    - prompt: 프롬프트 템플릿
    - llm: 모델 객체
    """
    return (
        {"context": retriever, "question": RunnablePassthrough()}  # 입력 데이터 처리
        | prompt  # 프롬프트 생성
        | llm  # 모델 응답 생성
        | MarkdownOutputParser()  # 마크다운 형식으로 결과 포맷팅
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


def process_input(llm, retriever, prompt, user_input):
    """
    입력된 질문과 문맥(Context)을 결합하여 모델에 전달.
    - llm: 모델 파이프라인 객체
    - retriever: 데이터베이스 검색 객체
    - prompt: ChatPromptTemplate 객체
    - user_input: 사용자 입력 질문
    """
    # 문맥(Context) 생성
    context = retriever.get_relevant_documents(user_input)
    context_text = " ".join(doc.page_content for doc in context) if context else "관련된 정보가 없습니다."
    
    # 프롬프트 생성
    input_text_with_context = prompt.format(context=context_text, question=user_input)
    
    # 모델 응답 생성
    response = llm.pipeline(input_text_with_context)
    
    # 응답에서 <|assistant|> 이후 텍스트 추출
    return extract_answer(response[0]['generated_text'])
