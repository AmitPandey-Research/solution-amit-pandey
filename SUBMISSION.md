# Project Documentation: RAG Bot for Specialized Question-Answering

## 1. Project Overview and Objectives

The primary objective of this project was to develop a Retrieval-Augmented Generation (RAG) Bot. This bot is designed to provide accurate answers by leveraging specialized agents that retrieve information from distinct, predefined data sources (Car Data and Country Data) or perform computations (Mathematical Agent).

The system aims to:
- Efficiently process and store structured (Car) and unstructured (Country) data.
- Intelligently route user queries to the most appropriate specialized agent.
- Utilize RAG principles to generate contextually relevant answers.
- Employ LangGraph for orchestrating the agentic workflow.
- Handle edge cases and queries outside the defined scopes gracefully.

## 2. Adherence to Requirements

This solution was built to meet the following core requirements:

### 2.1. Data Chunking & Storage
- **Implementation**: The provided data dumps (Car and Country) were processed and chunked into smaller, manageable units. This is crucial for efficient retrieval by the RAG agents.
- **Storage**: FAISS (Facebook AI Similarity Search) was chosen as the vector store. This allows for fast and scalable similarity searches on the embedded data chunks. Embeddings were generated using the `all-mpnet-base-v2` model from SentenceTransformers, known for its strong performance in semantic understanding.
- **Rationale**: FAISS provides a good balance of speed and resource usage for local vector database needs. `all-mpnet-base-v2` offers high-quality embeddings suitable for discerning semantic relationships in the data.

### 2.2. Retrieval-Augmented Generation (RAG)
- **Implementation**: Each specialized agent (Car and Country) incorporates RAG. When a query is received, the agent first retrieves relevant chunks from its dedicated vector store. This retrieved context is then used by a language model (via OpenAI API) to generate a final, informed answer.
- **Rationale**: RAG enhances the language model's responses by grounding them in specific, factual data, reducing hallucinations and improving accuracy for domain-specific questions.

### 2.3. Routing & Query Handling
- **Implementation**: A central `Router` component analyzes the user's query to determine its nature (car-related, country-related, mathematical, or other).
    - **Car Queries**: Routed to the `CarAgent`.
    - **Country Queries**: Routed to the `CountryAgent`.
    - **Mathematical Queries**: Routed to the `MathAgent`. This agent not only computes the result but is designed to verify its correctness.
    - **Out-of-Scope Queries**: If a query doesn't fit these categories, a predefined message is returned indicating the system cannot provide an answer.
- **Agent Invocation**: The system ensures only the relevant agent and its associated data are used for a given query, optimizing resource use and answer relevance.

### 2.4. Edge Cases & Error Handling
- **Implementation**:
    - **Missing Data**: If an agent cannot find relevant information in its data store, it's designed to communicate this rather than generating a speculative answer.
    - **Out-of-Scope Queries**: Handled by the router, as described above.
    - **API/System Errors**: Basic error handling is incorporated, such as checking for API key availability and vector store existence before proceeding.
- **Rationale**: Robust error handling is essential for a good user experience and system stability.

### 2.5. Use LangGraph for Agentic AI workflow
- **Implementation**: The entire multi-agent system is orchestrated using `LangGraph`. A `StateGraph` defines the nodes (agents, router) and edges (transitions based on routing decisions or agent outputs).
    - **State Management**: `GraphState` (a TypedDict) is used to maintain and pass information (query, history, intermediate results) between nodes in the graph.
    - **Conditional Edges**: LangGraph's conditional edges are used to direct the flow based on the router's output or intermediate results from agents (e.g., deciding if a math operation is needed after a car/country agent retrieves data).
- **Rationale**: LangGraph provides a powerful and flexible framework for building complex, stateful agentic applications, making it easier to manage the interactions between multiple specialized agents.

## 3. Key Technical Decisions and Trade-offs

- **Vector Store (FAISS)**:
    - *Decision*: Use FAISS for local vector storage.
    - *Pros*: High speed for similarity search, CPU version is accessible.
    - *Trade-offs*: Requires loading the index into memory, which could be a concern for extremely large datasets not encountered in this project's scope.

- **Embedding Model (`all-mpnet-base-v2`)**:
    - *Decision*: Use `all-mpnet-base-v2` for generating text embeddings.
    - *Pros*: Excellent semantic understanding, widely used and well-tested.
    - *Trade-offs*: Larger model size and higher one-time computation cost for embedding the datasets compared to smaller models, but beneficial for accuracy.

- **Multi-Agent Architecture with LangGraph**:
    - *Decision*: Employ distinct agents for cars, countries, and math, orchestrated by LangGraph.
    - *Pros*: Modularity, specialization, easier maintenance, and clear flow control via LangGraph.
    - *Trade-offs*: Increased initial setup complexity compared to a monolithic approach, and careful state management is required across graph transitions.

- **API Key Management**:
    - *Decision*: API keys are currently in a `constants.py` file.
    - *Pros*: Simple for development and local testing.
    - *Trade-offs*: Not secure for production. The documentation recommends moving to environment variables for a production deployment.

## 4. Project Structure and Components

- `src/main.py`: Contains the main orchestration logic, agent initialization, LangGraph setup, and the chat loop.
- `src/constants.py`: Stores configuration like API keys and vector database paths.
- `src/utilities/`: Directory for agent implementations:
    - `car_agent.py`: Handles car-related queries using RAG.
    - `country_agent.py`: Processes country-specific questions using RAG.
    - `math_agent.py`: Performs mathematical calculations and verifies results.
    - `code_agent.py`: A versatile agent that generates and executes Python scripts. It can be used for a variety of tasks, including:
        - Performing statistical analysis on the provided datasets (Car and Country data).
        - Generating distribution visualizations based on the data.
        - Handling general programming or code-related queries by creating and running appropriate Python scripts.
        Its primary mechanism involves using an LLM (GPT-3.5-turbo) to generate Python code based on the user's prompt, saving this code to a temporary script, and then executing it to produce results, which can include textual output (like statistical summaries) or trigger the creation of visualization files.
    - `router.py`: Analyzes user queries and directs them to the appropriate agent.
    - `memory.py`: Manages conversation history (though current implementation is basic).
- `data/vectordb/`: Stores the FAISS vector databases.
- `requirements.txt`: Lists project dependencies.

## 5. Potential Future Enhancements

- **Persistent Conversation History**: Implement saving and loading of conversation history for a more continuous user experience.
- **Advanced Error Verification**: Enhance the Math Agent's verification capabilities.
- **Asynchronous Operations**: For improved responsiveness, especially with multiple agent calls or I/O bound tasks.
- **Dynamic Agent Selection**: Explore more sophisticated routing mechanisms, potentially using an LLM for routing if complexity grows.

This documentation provides a detailed overview of the RAG Bot's design, implementation, and adherence to the project requirements. 