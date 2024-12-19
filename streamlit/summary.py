from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain.schema import HumanMessage, AIMessage
from model import llm
import uuid
import re

def extract_summary(history):
    # 리스트인 경우 문자열로 결합
    if isinstance(history, list):
        history = "\n".join(
            message.content for message in history if hasattr(message, "content")
        )

    # 정규식으로 모든 요약 추출
    matches = re.findall(
        r"Please generate an updated summary in Korean:\s*(.*?)(?=\n\s*Newly added conversation:|$)",
        history,
        re.DOTALL
    )
    
    if matches:
        # 각 요약을 마크다운 형식으로 합치기
        return "\n\n---\n\n".join(matches)
    return "요약이 없습니다."
