import os
from typing import TypedDict, List, Dict, Any, Callable, Optional

#from src.utilities.memory import LocalMemory 


from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document # For DB creation helper
from constants import EMBEDDING_MODEL_NAME_FOR_ROUTING_DB
# These examples are now primarily for the one-time DB creation.
ROUTING_EXAMPLES_FOR_DB_CREATION = [
    ("What is the horsepower of the Honda Civic?", "car_agent"),
    ("Tell me about India’s population.", "country_agent"),
    ("What is 56 * (45 + 2)?", "math_agent"),
    ("What's the capital of Canada?", "country_agent"),
    ("Find the fastest SUV.", "car_agent"),
    ("How many kilometers per liter does the vehicle give?", "car_agent"),
    ("Calculate the square root of 144", "math_agent"),
    ("What language is spoken in Brazil?", "country_agent"),
    ("Convert 10 USD to INR", "fallback"),
    ("Write a function to calculate the area of a circle", "code_agent"),
    ("Find the mean of all the ncap ratings of the cars", "car_math_agent"),
    ("What is the average price of Toyota cars?", "car_math_agent"),
    ("List all cars and their NCAP ratings, then find the maximum NCAP rating.", "car_math_agent"),
    ("Get the fuel efficiency of all BMWs and tell me the average.", "car_math_agent"),
    ("What is the sum of horsepower for all Audi cars listed?", "car_math_agent"),
]

ROUTING_VECTORDB_PATH = "../data/vectordb/routing_vectordb"


def _ensure_routing_vectorstore_exists():
    """Checks for the routing vector store and creates it if it doesn't exist."""
    if not os.path.exists(ROUTING_VECTORDB_PATH):
        print(f"Routing vector store not found at {ROUTING_VECTORDB_PATH}. Creating now...")
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(ROUTING_VECTORDB_PATH), exist_ok=True)
            
        embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME_FOR_ROUTING_DB)
        intent_docs = [
            Document(page_content=q, metadata={"label": label})
            for q, label in ROUTING_EXAMPLES_FOR_DB_CREATION
        ]
        try:
            vectorstore = FAISS.from_documents(intent_docs, embedding_model)
            vectorstore.save_local(ROUTING_VECTORDB_PATH)
            print(f"Routing vector store created and saved to {ROUTING_VECTORDB_PATH}")
        except Exception as e:
            print(f"FATAL: Error creating routing vector store: {e}")
            raise
    else:
        print(f"Routing vector store already exists at {ROUTING_VECTORDB_PATH}.")




    
    

    
    

def load_routing_vectorstore() -> FAISS:
    print("current working directory",os.getcwd())
    if not os.path.exists(ROUTING_VECTORDB_PATH):
        print(
            f"Routing vector store not found at {ROUTING_VECTORDB_PATH}. "
            f"Please ensure it's created, possibly by running the main script once."
            )

    _ensure_routing_vectorstore_exists()
    print(f"Loading routing vector store from {ROUTING_VECTORDB_PATH}")
    embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME_FOR_ROUTING_DB)
    return FAISS.load_local(ROUTING_VECTORDB_PATH, embedding_model, allow_dangerous_deserialization=True)
    