import os
print("current directory",os.getcwd())
from typing import TypedDict, List, Dict, Any, Callable, Optional
from langgraph.graph import StateGraph, END
from constants import (
    ACCESS,
    EMBEDDING_MODEL_NAME_FOR_ROUTING_DB,
    CAR_VECTORDB_PATH,
    COUNTRY_VECTORDB_PATH
)
EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_NAME_FOR_ROUTING_DB

from utilities.router import Router
from utilities.car_agent import CarAgent 
from utilities.country_agent import CountryAgent 
from utilities.math_agent import run_math_agent 
from utilities.code_agent import run_code_agent 
from utilities.memory import LocalMemory
from utilities.query_refiner import refine_query 
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI 

# --- State Definition for the Graph ---
class GraphState(TypedDict):
    query: str                 
    original_query: str       
    result: str                
    history: List[Dict[str, str]] 
    agent_scratchpad: List[Any] 
    next_agent_operation: Optional[str] 
    intermediate_data_for_math: Optional[str] 


# --- Main Orchestration Logic ---
def agent_orchestrator_chat():
    """
    Initializes agents, the router, memory, and the LangGraph workflow,
    then runs the chat loop.
    """
    if not ACCESS:
        print("OpenAI API Key is not configured. Exiting.")
        return

    # --- 1. Initialize Embedding Model (shared for Car and Country agents) ---
    try:
        embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Error initializing embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        print("Please ensure the model is available or install sentence-transformers.")
        return

    # --- 2. Load Vector Databases for Car and Country Agents ---
    if not os.path.exists(CAR_VECTORDB_PATH):
        print(f"Car agent vector database not found at: {CAR_VECTORDB_PATH}")
        print("Please ensure it is created and populated before running.")
        return
    if not os.path.exists(COUNTRY_VECTORDB_PATH):
        print(f"Country agent vector database not found at: {COUNTRY_VECTORDB_PATH}")
        print("Please ensure it is created and populated before running.")
        return

    try:
        car_vector_db = FAISS.load_local(
            CAR_VECTORDB_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        print(f"Car vector DB loaded successfully from {CAR_VECTORDB_PATH}")

        country_vector_db = FAISS.load_local(
            COUNTRY_VECTORDB_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        print(f"Country vector DB loaded successfully from {COUNTRY_VECTORDB_PATH}")
    except Exception as e:
        print(f"Error loading agent vector databases: {e}")
        return

    
    try:
        car_agent = CarAgent(vector_store=car_vector_db) 
        country_agent = CountryAgent(vector_store=country_vector_db)
    except NameError: 
        print("ERROR: CarAgent or CountryAgent class not found. Ensure they are defined in utilities.")
        return
    except Exception as e:
        print(f"Error initializing CarAgent or CountryAgent: {e}")
        return
    try:
        router = Router()
        print("Router initialized successfully.")
    except Exception as e:
        print(f"Error initializing Router: {e}")
        return

    # --- 5. Initialize Memory ---
    memory = LocalMemory()

    # --- 6. Create Agent Workflow Graph ---
    workflow = StateGraph(GraphState)

    # --- Define Nodes ---
    def route_query_node(state: GraphState) -> Dict[str, Any]:
        print(f"\n--- Routing Query ---")
        query = state["query"]
        history = state["history"]
        context = "\n".join([f"User: {h['query']}\nAI: {h['response']}" for h in history]) # Adjusted key for response
        refined_query = query
        print(f"Original Query: {query}")
        print(f"Refined Query: {refined_query}")

        route_decision = router.route(refined_query)
        print(f"Route Decision: {route_decision}")

        return {
            "query": refined_query, 
            "original_query": query, 
            "next_agent_operation": route_decision 
        }

    def car_agent_node(state: GraphState) -> Dict[str, Any]:
        print(f"\n--- Car Agent Node ---")
        query = state["query"]
        result = car_agent.run(query) 
        print(f"Car Agent Result: {result[:200]}...") 

        if state.get("next_agent_operation") == "car_math_agent":
            return {
                "intermediate_data_for_math": result,
                "next_agent_operation": "math_on_car_output" 
            }
        return {"result": result}


    def country_agent_node(state: GraphState) -> Dict[str, Any]:
        print(f"\n--- Country Agent Node ---")
        query = state["query"]
        result = country_agent.run(query)
        print(f"Country Agent Result: {result[:200]}...")

        if state.get("next_agent_operation") == "country_math_agent":
            return {
                "intermediate_data_for_math": result,
                "next_agent_operation": "math_on_country_output"
            }
        return {"result": result}

    def math_agent_node(state: GraphState) -> Dict[str, Any]:
        print(f"\n--- Math Agent Node ---")
        query_for_math = state["query"] 

        if state.get("next_agent_operation") == "math_on_car_output" or \
           state.get("next_agent_operation") == "math_on_country_output":
            original_intent_query = state["original_query"]
            intermediate_data = state.get("intermediate_data_for_math", "")
            query_for_math = (
                f"Based on the following data: '{intermediate_data}'. "
                f"And considering the original request: '{original_intent_query}'. "
                f"Perform the necessary mathematical calculations."
            )
            print(f"Query for Math (derived): {query_for_math}")
        else:
            print(f"Query for Math (direct): {query_for_math}")

        result = run_math_agent(query_for_math)
        print(f"Math Agent Result: {result}")
        return {"result": result}

    def code_agent_node(state: GraphState) -> Dict[str, Any]:
        print(f"\n--- Code Agent Node ---")
        query = state["query"]
        result = run_code_agent(query)
        print(f"Code Agent Result: {result}")
        return {"result": result}

    def fallback_node(state: GraphState) -> Dict[str, Any]:
        print(f"\n--- Fallback Node ---")
        return {"result": "I'm sorry, I cannot answer that question with my current capabilities."}

    # --- Add Nodes to Workflow ---
    workflow.add_node("router", route_query_node)
    workflow.add_node("car_agent", car_agent_node)
    workflow.add_node("country_agent", country_agent_node)
    workflow.add_node("math_agent", math_agent_node)
    workflow.add_node("code_agent", code_agent_node)
    workflow.add_node("fallback", fallback_node)

    # --- Define Edges ---
    workflow.set_entry_point("router")

    def decide_next_node(state: GraphState) -> str:
        operation = state.get("next_agent_operation")
        if operation == "car_agent" or operation == "car_math_agent":
            return "car_agent"
        elif operation == "country_agent" or operation == "country_math_agent":
            return "country_agent"
        elif operation == "math_agent": 
            return "math_agent"
        elif operation == "code_agent":
            return "code_agent"
        elif operation == "math_on_car_output" or operation == "math_on_country_output":
            return "math_agent"
        return "fallback" 

    workflow.add_conditional_edges(
        "router",
        decide_next_node,
        {
            "car_agent": "car_agent",
            "country_agent": "country_agent",
            "math_agent": "math_agent",
            "code_agent": "code_agent",
            "fallback": "fallback",
        }
    )

    # Edges from Car Agent
    def after_car_agent_decision(state: GraphState) -> str:
        if state.get("next_agent_operation") == "math_on_car_output":
            return "math_agent" 
        return END 

    workflow.add_conditional_edges("car_agent", after_car_agent_decision, {
        "math_agent": "math_agent",
        END: END
    })

    # Edges from Country Agent
    def after_country_agent_decision(state: GraphState) -> str:
        if state.get("next_agent_operation") == "math_on_country_output":
            return "math_agent" 
        return END

    workflow.add_conditional_edges("country_agent", after_country_agent_decision, {
        "math_agent": "math_agent",
        END: END
    })

    # Simple agents go to END
    workflow.add_edge("math_agent", END)
    workflow.add_edge("code_agent", END)
    workflow.add_edge("fallback", END)

    # --- Compile Workflow ---
    app = workflow.compile()
    print("\nAgent workflow compiled successfully.")

    # --- 7. Run Chat Loop ---
    print("\n--- RAG Agent Chatbot Initialized (type 'exit' to quit) ---")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chatbot...")
            break
        if not user_input.strip():
            continue

        current_history = memory.load_memory_variables({}).get("history", [])
        initial_graph_state: GraphState = {
            "query": user_input,
            "original_query": user_input,
            "history": current_history,
            "result": "",
            "agent_scratchpad": [],
            "next_agent_operation": None,
            "intermediate_data_for_math": None
        }

        try:
            # Stream events for more detailed logging
            events = []
            print("--- Invoking Workflow ---")
            for event_part in app.stream(initial_graph_state):
                events.append(event_part)
            
            final_answer = "Could not determine a final answer."
            for i in range(len(events) - 1, -1, -1):
                event_output = events[i]
                for node_name, node_output_dict in event_output.items():
                    if isinstance(node_output_dict, dict) and "result" in node_output_dict and node_output_dict["result"]:
                        final_answer = node_output_dict["result"]
                        break 
                if final_answer != "Could not determine a final answer.":
                    break
            
            print(f"AI: {final_answer}")
            memory.save_context({"query": user_input}, {"response": final_answer}) 
        except Exception as e:
            print(f"Error during workflow execution: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    agent_orchestrator_chat()
