from langchain.chains.llm_math.base import LLMMathChain
from langchain_community.chat_models import ChatOpenAI
from typing import Callable
from constants import ACCESS


def create_math_agent_chain() -> LLMMathChain:
    llm = ChatOpenAI(temperature=0, openai_api_key=ACCESS)
    return LLMMathChain.from_llm(llm=llm, verbose=True)


def run_math_agent(prompt: str) -> str:
    """Runs the math agent using the LLMMathChain."""
    math_chain = create_math_agent_chain()
    return math_chain.invoke({"question": prompt})  # use invoke() with a dict input
