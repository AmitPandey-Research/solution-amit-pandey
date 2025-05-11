import constants
import os
import re
import pandas as pd
from typing import List, Dict, Any

from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, LLMMathChain


from langgraph.graph import StateGraph, END
from typing import TypedDict


os.environ["OPENAI_API_KEY"] = constants.API_KEY


def parse_engine_specs(engine_str: str) -> Dict[str, Any]:
    engine_match = re.match(r"(\w+),\s*(\d+)\s*HP,\s*(\d+)\s*cc", engine_str, re.IGNORECASE)
    if engine_match:
        engine_type, hp, cc = engine_match.groups()
        return {
            "engine_type": engine_type,
            "horse_power": int(hp),
            "cc": int(cc)
        }
    return {}

def parse_other_specs(specs_str: str) -> Dict[str, Any]:
    body_type_match = re.match(r"(\w+)", specs_str)
    mileage_match = re.search(r"(\d+)\s*km/l", specs_str)
    speed_match = re.search(r"(\d+)\s*km/h", specs_str)
    
    return {
        "body_type": body_type_match.group(1) if body_type_match else None,
        "mileage_kmpl": int(mileage_match.group(1)) if mileage_match else None,
        "top_speed_kmph": int(speed_match.group(1)) if speed_match else None
    }

def load_documents(file_path: str) -> List[Document]:
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        cars_json = []
        for _, row in df.iterrows():
            car_data = {
                "car_name": row.get("Car Name"),
                "manufacturer": row.get("Manufacturer"),
                "launch_year": row.get("Launch Year"),
                "description": row.get("Description"),
                "engine": parse_engine_specs(row.get("Engine Specifications", "")),
                "specifications": parse_other_specs(row.get("Other Specifications", "")),
                "user_ratings": float(row.get("User Ratings", 0)),
                "ncap_rating": int(row.get("NCAP Global Rating", 0))
            }
            cars_json.append(car_data)
        docs = [Document(page_content=str(cars_json))]
    else:
        loader = TextLoader(file_path)
        docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


def create_vector_store(docs):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

def create_rag_bot(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Cell 5: Load documents and create agents
country_docs = load_documents("../data/country_data.md")
car_docs = load_documents("../data/cars_dataset.csv")

country_vector = create_vector_store(country_docs)
car_vector = create_vector_store(car_docs)

car_agent = create_rag_bot(car_vector)
country_agent = create_rag_bot(country_vector)
math_agent = LLMMathChain(llm=ChatOpenAI(temperature=0), verbose=True)


def car_agent_node(state):
    return {"result": car_agent.invoke({"query": state["query"]})["result"]}

def country_agent_node(state):
    return {"result": country_agent.invoke({"query": state["query"]})["result"]}

def math_agent_node(state):
    return {"result": math_agent.run(state["query"])}

def fallback_node(state):
    return {"result": "Sorry, I cannot answer that question."}

def router_node(state):
    query = state["query"].lower()
    if any(word in query for word in ["car", "engine", "vehicle", "ncap", "mileage", "manufacturer"]):
        return {"next": "car_agent"}
    elif any(word in query for word in ["country", "capital", "population", "continent", "language"]):
        return {"next": "country_agent"}
    elif any(op in query for op in ["+", "-", "*", "/", "sqrt", "^", "what is", "calculate"]):
        return {"next": "math_agent"}
    else:
        return {"next": "fallback"}


class GraphState(TypedDict):
    query: str
    result: str



graph = StateGraph(GraphState)

graph.add_node("router", router_node)
graph.add_node("car_agent", car_agent_node)
graph.add_node("country_agent", country_agent_node)
graph.add_node("math_agent", math_agent_node)
graph.add_node("fallback", fallback_node)

graph.set_entry_point("router")
graph.add_conditional_edges("router", lambda state: state["next"], {
    "car_agent": "car_agent",
    "country_agent": "country_agent",
    "math_agent": "math_agent",
    "fallback": "fallback"
})

graph.add_edge("car_agent", END)
graph.add_edge("country_agent", END)
graph.add_edge("math_agent", END)
graph.add_edge("fallback", END)

rag_workflow = graph.compile()


def agent_orchestrator_chat():
    print("Welcome to the RAG Bot (type 'exit' to quit)")
    while True:
        query = input("Q: ")
        if query.lower() == "exit":
            break
        response = rag_workflow.invoke({"query": query})
        print("A:", response["result"])



if __name__ == "__main__":
    agent_orchestrator_chat()

