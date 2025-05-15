# import os
# from langchain.chains import RetrievalQA
# from langchain_openai import ChatOpenAI
# from langchain_community.vectorstores import FAISS
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from typing import Dict, Any, List
# import re
# from langchain.prompts import PromptTemplate
# from constants import ACCESS
# class CarAgent:
#     def __init__(self, vector_store: FAISS):
#         self.vector_store = vector_store
        
#         # Check for OpenAI API key
#         openai_api_key = ACCESS
#         if not openai_api_key:
#             raise ValueError(
#                 "OpenAI API key not found! Please set the OPENAI_API_KEY environment variable."
#                 "\nYou can get an API key from https://platform.openai.com/api-keys"
#             )
            
#         self.llm = ChatOpenAI(
#             model_name="gpt-3.5-turbo",
#             temperature=0,
#             openai_api_key=openai_api_key
#         )

#         self.prompt = PromptTemplate.from_template("""
#             You are a car expert.  Use the provided context to answer the user's question about cars.
#             Context:
#             {context}

#             Question:
#             {question}

#             Answer:
#             """)
#         self.chain = RetrievalQA.from_chain_type(
#             llm=self.llm,
#             retriever=self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10}),
#             return_source_documents=False,
#             prompt = self.prompt
#         )

#     def get_relevant_cars(self, query: str) -> List[Dict[str, Any]]:
#         """
#         Retrieves relevant car information from the vector store based on the query.
#         """
#         return self.vector_store.similarity_search_with_relevance_scores(query, k=5)

#     def extract_filter_criteria(self, query: str) -> Dict[str, Any]:
#         """
#         Extracts filter criteria (e.g., make, model, year) from the query.  This is a placeholder;
#         a real implementation would use a more sophisticated method (e.g., a language model).
#         """
#         filters = {}
#         if "make" in query.lower():
#             # Modified regex to capture multiple words after "make"
#             make_match = re.search(r"make\s+([a-zA-Z\s]+)", query.lower())
#             if make_match:
#                 filters["manufacturer"] = make_match.group(1).strip().title() # Capture, trim whitespace, and title case
#         if "model" in query.lower():
#             model_match = re.search(r"model\s+([a-zA-Z0-9\s]+)", query.lower()) # Allow spaces in model names
#             if model_match:
#                 filters["car_name"] = model_match.group(1).strip().title()
#         if "year" in query.lower():
#              year_match = re.search(r"year\s+(\d{4})", query.lower())
#              if year_match:
#                 filters["launch_year"] = int(year_match.group(1))
#         return filters
    
#     def refine_query(self, query: str, context: str) -> str:
#         """Refines the query based on the context.  For the car agent, we assume the query is fine."""
#         return query

#     def run(self, query: str) -> str:
#         """
#         Runs the Car Agent to answer a query.
#         """
#         # 1. Refine the query
#         refined_query = self.refine_query(query, "") # No context for first query
#         print("refined query given to the car agent",refined_query)

#         # 2. Extract filter criteria
#         filters = self.extract_filter_criteria(refined_query)
#         print("extracted_filters",filters)

#         # 3. Retrieve relevant documents, applying filters
#         if filters:
#             results = self.vector_store.similarity_search(refined_query, filter=filters)
#             if results == []:
#                 results = self.vector_store.similarity_search(refined_query)
#         else:
#             results = self.vector_store.similarity_search(refined_query)
#         print("retrieved car docs:",results)
#         # 4. Format the results
#         context = "\n".join([doc.page_content for doc in results])

#         print("vector db car agent retrieved context",context)
        
#         # 5. Run chain
#         answer = self.chain.invoke({"query": refined_query, "context": context})
#         return answer["result"]


#---------------------------------------------------

import os
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from typing import Dict, Any, List
import re
from langchain.prompts import PromptTemplate
from constants import ACCESS

class CarAgent:
    def __init__(self, vector_store: FAISS):
        self.vector_store = vector_store

        # Check for OpenAI API key
        openai_api_key = ACCESS
        if not openai_api_key:
            raise ValueError(
                "OpenAI API key not found! Please set the OPENAI_API_KEY environment variable."
                
            )

        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=openai_api_key
        )

        self.prompt = PromptTemplate.from_template("""
            You are a car expert. Use the provided context to answer the user's question about cars.
            Context:
            {context}

            Question:
            {question}

            Answer:
            """)
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10}),
            return_source_documents=False,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def extract_filter_criteria(self, query: str) -> Dict[str, Any]:
        """
        Extracts filter criteria (e.g., make, model, year) from the query.
        """
        filters = {}
        if "make" in query.lower():
            make_match = re.search(r"make\s+([a-zA-Z0-9\s\-\.]+)", query)
            if make_match:
                filters["manufacturer"] = make_match.group(1).strip()
        if "model" in query.lower():
            model_match = re.search(r"model\s+([a-zA-Z0-9\s\-\.]+)", query)
            if model_match:
                filters["car_name"] = model_match.group(1).strip()
        if "year" in query.lower():
            year_match = re.search(r"year\s+(\d{4})", query)
            if year_match:
                filters["launch_year"] = int(year_match.group(1))
        return filters

    def refine_query(self, query: str, context: str) -> str:
        """Refines the query based on the context. For the car agent, we assume the query is fine."""
        return query

    def run(self, query: str) -> str:
        """
        Runs the Car Agent to answer a query.
        """
        # 1. Refine the query
        refined_query = self.refine_query(query, "") # No context for first query
        print("refined query given to the car agent", refined_query)

        # 2. Extract filter criteria
        filters = self.extract_filter_criteria(refined_query)
        print("extracted_filters", filters)

        # 3. Retrieve relevant documents, applying filters
        if filters:
            results = self.vector_store.similarity_search(refined_query, filter=filters)
            print("retrieved car docs (with filters):", results)
            if results == []:
              results = self.vector_store.similarity_search(refined_query) 
              print("retrieved car docs (with filters but no result so still):", results) 
        else:
            results = self.vector_store.similarity_search(refined_query)
            print("retrieved car docs (no filters):", results)

        # 4. Format the results into context
        context = "\n".join([doc.page_content for doc in results])
        print("vector db car agent retrieved context:\n", context)

        # 5. Run chain with the retrieved context
        answer = self.chain.invoke({"query": refined_query, "context": context})
        print("Car Agent Raw Answer:", answer)
        return answer["result"]