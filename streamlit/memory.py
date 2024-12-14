from langchain.memory import ConversationSummaryMemory

# 세션 기록을 저장할 딕셔너리
store = {}

def create_summary_memory(session_id, llm):
    """
    세션 ID와 EXAONE LLM을 기반으로 ConversationSummaryMemory 생성.
    """
    if session_id not in store:
        # 새 메모리 생성
        store[session_id] = ConversationSummaryMemory(llm=llm, return_messages=True)
    return store[session_id]
