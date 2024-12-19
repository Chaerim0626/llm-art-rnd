from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain.schema import HumanMessage, AIMessage
from model import llm
import uuid

# Custom Prompt Template
summary_prompt = PromptTemplate(
    input_variables=["summary", "new_lines"],
    template="""
    Previous conversation summary:
    {summary}

    Newly added conversation:
    {new_lines}

    Please generate an updated summary in Korean:
    """
)
memory = ConversationSummaryMemory(
    llm=llm,
    max_token_limit=500,  
    return_messages=True,
    memory_key="history"
)
