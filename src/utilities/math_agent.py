from langchain.chains.llm_math.base import LLMMathChain
from langchain_community.chat_models import ChatOpenAI
from typing import Callable
import subprocess
import os
import hashlib
from openai import OpenAI
from constants import ACCESS

client = OpenAI(api_key=ACCESS)


def create_math_agent_chain() -> LLMMathChain:
    llm = ChatOpenAI(temperature=0)
    return LLMMathChain(llm=llm, verbose=True)



def run_math_agent(prompt: str) -> str:
    """Runs the math agent using the LLMMathChain."""
    math_chain = create_math_agent_chain()
    return math_chain.run(prompt)