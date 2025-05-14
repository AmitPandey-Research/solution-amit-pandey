# src/main.py
import os
print("current directory",os.getcwd())
from typing import TypedDict, List, Dict, Any, Callable, Optional

from langgraph.graph import StateGraph, END

# Project-specific imports from src
from constants import (
    ACCESS,
    EMBEDDING_MODEL_NAME_FOR_ROUTING_DB,
    CAR_VECTORDB_PATH,
    COUNTRY_VECTORDB_PATH,
    #LLM_MODEL_NAME_AGENTS
)
EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_NAME_FOR_ROUTING_DB

from utilities.router import Router
from utilities.car_agent import CarAgent # Assuming CarAgent class exists
from utilities.country_agent import CountryAgent # Assuming CountryAgent class exists
from utilities.math_agent import run_math_agent # Assuming a callable function
from utilities.code_agent import run_code_agent # Assuming a callable function
from utilities.memory import LocalMemory
from utilities.query_refiner import refine_query # General query refiner

# Langchain Community imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI # For agents if they use a separate LLM

# --- State Definition for the Graph ---
class GraphState(TypedDict):
    query: str                 # The current query to process (can be refined)
    original_query: str        # The initial query from the user
    result: str                # The final result from an agent or the graph
    history: List[Dict[str, str]] # Conversation history
    agent_scratchpad: List[Any] # For agents that use scratchpads (e.g., ReAct)

    # Control flags for multi-step agent flows
    next_agent_operation: Optional[str] # e.g., 'math_on_car_output', 'math_on_country_output'
    intermediate_data_for_math: Optional[str] # Data from car/country agent for math agent


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

    # --- 3. Initialize Agents ---
    # Assuming agents might use a different LLM or have specific configurations
    # agent_llm = ChatOpenAI(model_name=LLM_MODEL_NAME_AGENTS, openai_api_key=OPENAI_API_KEY, temperature=0)

    # Placeholder: Actual CarAgent and CountryAgent might take LLM, tools, etc.
    # For this example, they primarily need their vector DBs for retrieval.
    # Their internal run() method would handle the RAG logic.
    try:
        car_agent = CarAgent(vector_store=car_vector_db) # Pass the loaded FAISS instance
        country_agent = CountryAgent(vector_store=country_vector_db)
    except NameError: # If CarAgent/CountryAgent classes are not defined
        print("ERROR: CarAgent or CountryAgent class not found. Ensure they are defined in utilities.")
        return
    except Exception as e:
        print(f"Error initializing CarAgent or CountryAgent: {e}")
        return


    # --- 4. Initialize Router ---
    # The router loads its own vector DB as defined in its class and routing_vectordb.py
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
        #refined_query = refine_query(query=query, context=context)
        print(f"Original Query: {query}")
        print(f"Refined Query: {refined_query}")

        route_decision = router.route(refined_query)
        print(f"Route Decision: {route_decision}")

        return {
            "query": refined_query, # Update state with refined query
            "original_query": query, # Keep original query for context
            "next_agent_operation": route_decision # This will guide conditional edges
        }

    def car_agent_node(state: GraphState) -> Dict[str, Any]:
        print(f"\n--- Car Agent Node ---")
        query = state["query"]
        result = car_agent.run(query) # Assuming CarAgent has a run method
        print(f"Car Agent Result: {result[:200]}...") # Print snippet

        if state.get("next_agent_operation") == "car_math_agent":
            return {
                "intermediate_data_for_math": result,
                "next_agent_operation": "math_on_car_output" # Signal to proceed to math
            }
        return {"result": result}


    def country_agent_node(state: GraphState) -> Dict[str, Any]:
        print(f"\n--- Country Agent Node ---")
        query = state["query"]
        result = country_agent.run(query) # Assuming CountryAgent has a run method
        print(f"Country Agent Result: {result[:200]}...")

        if state.get("next_agent_operation") == "country_math_agent":
            return {
                "intermediate_data_for_math": result,
                "next_agent_operation": "math_on_country_output" # Signal to proceed to math
            }
        return {"result": result}

    def math_agent_node(state: GraphState) -> Dict[str, Any]:
        print(f"\n--- Math Agent Node ---")
        query_for_math = state["query"] # Default to current query

        if state.get("next_agent_operation") == "math_on_car_output" or \
           state.get("next_agent_operation") == "math_on_country_output":
            original_intent_query = state["original_query"]
            intermediate_data = state.get("intermediate_data_for_math", "")
            # Formulate a new query for the math agent. This might need an LLM call for robustness.
            # For now, a simple concatenation or instruction.
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
        elif operation == "math_agent": # Direct math query
            return "math_agent"
        elif operation == "code_agent":
            return "code_agent"
        # For multi-step operations where data is now ready for math
        elif operation == "math_on_car_output" or operation == "math_on_country_output":
            return "math_agent"
        return "fallback" # Default to fallback

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
            return "math_agent" # Proceed to math agent
        return END # End of operation for simple car_agent query

    workflow.add_conditional_edges("car_agent", after_car_agent_decision, {
        "math_agent": "math_agent",
        END: END
    })

    # Edges from Country Agent
    def after_country_agent_decision(state: GraphState) -> str:
        if state.get("next_agent_operation") == "math_on_country_output":
            return "math_agent" # Proceed to math agent
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
                # print(f"Event: {event_part}") # Detailed event logging
            
            # The final result should be in the 'result' field of the last relevant state update
            final_answer = "Could not determine a final answer."
            # Iterate backwards through events to find the last node that produced a 'result'
            for i in range(len(events) - 1, -1, -1):
                event_output = events[i]
                # Event output is a dict like {'node_name': {'field': value, ...}}
                for node_name, node_output_dict in event_output.items():
                    if isinstance(node_output_dict, dict) and "result" in node_output_dict and node_output_dict["result"]:
                        final_answer = node_output_dict["result"]
                        break 
                if final_answer != "Could not determine a final answer.":
                    break
            
            print(f"AI: {final_answer}")
            memory.save_context({"query": user_input}, {"response": final_answer}) # Adjusted key for response

        except Exception as e:
            print(f"Error during workflow execution: {e}")
            # Optionally, log the full error stack trace for debugging
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    agent_orchestrator_chat()











#----------------------------------------------------------------------------------------------------------------

# from typing import TypedDict, Dict, Any, List
# from langgraph.graph import StateGraph, END
# from src.utilities.dataloader import DataLoader
# from src.utilities.code_agent import run_code_agent
# from src.utilities.math_agent import run_math_agent
# from src.utilities.car_agent import CarAgent
# from src.utilities.country_agent import CountryAgent
# from src.utilities.memory import LocalMemory
# from src.utilities.query_refiner import refine_query
# from src.utilities.router import Router


# class GraphState(TypedDict):
#     query: str
#     result: str
#     history: List[Dict[str, str]]


# def create_agent_workflow(
#     car_agent: CarAgent, country_agent: CountryAgent, math_agent: Callable, memory: BaseMemory
# ) -> StateGraph:
#     """Creates the LangGraph workflow for the agent system."""

#     def router_node(state: GraphState) -> str:
#         """Routes the query to the appropriate agent."""
#         query = state["query"]
#         history = state["history"]

#         # Get previous context
#         context = "\n".join([f"Q: {h['query']} A: {h['response']}" for h in history])

#         # Refine query based on history
#         refined_query = refine_query(query, context)  # Use general refine_query

#         router = Router(routing_examples=ROUTING_EXAMPLES) # Initialize Router
#         agent = router.route(refined_query)
#         print(f"Routing to {agent}")
#         return agent

#     def car_agent_node(state: GraphState) -> Dict[str, Any]:
#         """Handles car-related queries."""
#         query = state["query"]
#         history = state["history"]
#         context = "\n".join([f"Q: {h['query']} A: {h['response']}" for h in history])

#         # Refine the query for the car agent
#         refined_query = car_agent.refine_query(query, context)

#         result = car_agent.run(refined_query)
#         return {"result": result}

#     def country_agent_node(state: GraphState) -> Dict[str, Any]:
#         """Handles country-related queries."""
#         query = state["query"]
#         history = state["history"]
#         context = "\n".join([f"Q: {h['query']} A: {h['response']}" for h in history])
#         refined_query = country_agent.refine_query(query, context)
#         result = country_agent.run(refined_query)
#         return {"result": result}

#     def math_agent_node(state: GraphState) -> Dict[str, Any]:
#         """Handles math-related queries."""
#         query = state["query"]
#         result = run_math_agent(query, method="llm")  # Or "exec"
#         return {"result": result}

#     def fallback_node(state: GraphState) -> Dict[str, Any]:
#         """Handles queries that cannot be routed."""
#         return {"result": "Sorry, I cannot answer that question."}

#     graph = StateGraph(GraphState)
#     graph.add_node("router", router_node)
#     graph.add_node("car_agent", car_agent_node)
#     graph.add_node("country_agent", country_agent_node)
#     graph.add_node("math_agent", math_agent_node)
#     graph.add_node("fallback", fallback_node)

#     graph.set_entry_point("router")
#     graph.add_conditional_edges(
#         "router",
#         lambda state: state["next"],
#         {
#             "car_agent": "car_agent",
#             "country_agent": "country_agent",
#             "math_agent": "math_agent",
#             "fallback": "fallback",
#         },
#     )

#     graph.add_edge("car_agent", END)
#     graph.add_edge("country_agent", END)
#     graph.add_edge("math_agent", END)
#     graph.add_edge("fallback", END)
#     return graph.compile()


# def agent_orchestrator_chat():
#     """Main function to run the agent orchestrator."""
#     # Load data and create agents
#     country_docs = DataLoader.load_documents("data/country_data.md")
#     car_docs = DataLoader.load_documents("data/cars_dataset.csv")

#     country_vector = FAISS.from_documents(country_docs, SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))
#     car_vector = FAISS.from_documents(car_docs, SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))

#     car_agent = CarAgent(car_vector)
#     country_agent = CountryAgent(country_vector)
#     math_agent = run_math_agent  # Use the function directly
#     memory = LocalMemory()
#     rag_workflow = create_agent_workflow(car_agent, country_agent, math_agent, memory)

#     print("Welcome to the RAG Bot (type 'exit' to quit)")
#     while True:
#         query = input("Q: ")
#         if query.lower() == "exit":
#             break
#         response = rag_workflow.invoke({"query": query, "history": memory.load_memory_variables({})["history"]})
#         memory.save_context({"query": query}, {"result": response["result"]})
#         print("A:", response["result"])



# # Global routing examples
# ROUTING_EXAMPLES = [
#     ("What is the horsepower of the Honda Civic?", "car_agent"),
#     ("Tell me about India’s population.", "country_agent"),
#     ("What is 56 * (45 + 2)?", "math_agent"),
#     ("What's the capital of Canada?", "country_agent"),
#     ("Find the fastest SUV.", "car_agent"),
#     ("How many kilometers per liter does the vehicle give?", "car_agent"),
#     ("Calculate the square root of 144", "math_agent"),
#     ("What language is spoken in Brazil?", "country_agent"),
#     ("Convert 10 USD to INR", "fallback"),
# ]


# if __name__ == "__main__":
#     agent_orchestrator_chat()


# # import constants
# # import os
# # import sys
# # import glob
# # from utilities.dataloader import DataLoader

# # from langchain_community.chat_models import ChatOpenAI
# # from langchain_community.vectorstores import FAISS
# # from langchain.chains import RetrievalQA, LLMMathChain


# # os.environ["OPENAI_API_KEY"] = constants.ACCESS


# # if __name__ == "__main__":
# #     if len(sys.argv) < 2:
# #         print("Error: Please provide a command. Choose from create_db or chat!")
# #         sys.exit(1)

# #     command = sys.argv[1]
# #     if command == "create_db":
# #         files = glob.glob(r"C:\Users\infinix\OneDrive\Desktop\Amit_Pandey\solution-amit-pandey\data\raw\*")
# #         if len(files) == 0:
# #             print("length zero")
# #         for file in files:
# #             print(">>>File:", file)
# #             if os.path.isfile(file):
# #                 print(f"[INFO] Indexing {file}")
# #                 DataLoader.create_index(file)

# #     if command == "chat":
# #         print("-----CHAT STARTED-----")
# #         #agent_orchestrator_chat()  # Enable when ready

