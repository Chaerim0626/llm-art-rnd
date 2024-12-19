import re
import yaml
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain.chains import ConversationChain
from operator import itemgetter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

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


def create_chain(retriever, prompt, llm):
    """
    컨텍스트와 대화 이력을 결합해 모델 체인 구성.
    - retriever: 작품 관련 문서 검색 객체
    - prompt: ChatPromptTemplate 객체
    - llm: 언어 모델 객체
    """
    chain = RunnableMap({
        "context": lambda inputs: retriever.get_relevant_documents(inputs["question"]),
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }) | prompt | llm | MarkdownOutputParser()

    return chain



def extract_answer(text):
    """
    응답 텍스트에서 'Answer:' 또는 특정 패턴 이후 텍스트를 추출.
    - text: 모델에서 생성된 텍스트
    """
    match = re.search(r"<\|assistant\|>\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "답변을 찾을 수 없습니다."
