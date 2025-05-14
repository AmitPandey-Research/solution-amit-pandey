from langchain_core.memory import BaseMemory
from typing import List, Dict, Any

class LocalMemory(BaseMemory):
    """
    Local memory that stores previous queries and responses.
    """
    history: List[Dict[str, str]] = []

    @property
    def memory_keys(self) -> List[str]:
        return ["history"]

    @property
    def memory_variables(self) -> List[str]:
        return ["history"]

    def clear(self) -> None:
        self.history = []

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"history": self.history}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        self.history.append({"query": inputs.get("query", ""), "response": outputs.get("result", "")})