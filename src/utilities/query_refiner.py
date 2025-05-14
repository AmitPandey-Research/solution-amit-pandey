from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from constants import ACCESS

def refine_query(query: str, context: str) -> str:
    """Refines the query based on the context."""
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,openai_api_key=ACCESS)
    refine_prompt_template = """
        Given the previous conversation and the current question, refine the current question to be a standalone question.

        Previous Conversation:
        {context}

        Current Question:
        {question}

        Standalone Question:
        """
    refine_prompt = PromptTemplate.from_template(refine_prompt_template)
    refine_chain = refine_prompt | llm | StrOutputParser()
    return refine_chain.invoke({"question": query, "context": context})