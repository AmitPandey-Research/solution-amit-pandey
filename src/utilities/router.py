import os
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from constants import ACCESS 
from utilities.routing_vectordb import load_routing_vectorstore

class Router:
    def __init__(
        self,
        llm_model_name: str = "gpt-3.5-turbo", 
    ):
        openai_api_key = ACCESS
        if not openai_api_key:
            raise ValueError(
                "OpenAI API key (ACCESS) not found in constants! Please set it."
                "\nYou can get an API key from https://platform.openai.com/api-keys"
            )
            
        self.llm = ChatOpenAI(
            model_name=llm_model_name,
            temperature=0,
            openai_api_key=openai_api_key
        )
        
        self.routing_vectorstore = load_routing_vectorstore()
        print("routing vectorstore",self.routing_vectorstore)

    def route(self, query: str) -> str:
        print("query inside router",query)
        top_k = 3 
        filtered_docs = []
        seen_labels = set()

        retrieved_docs = self.routing_vectorstore.similarity_search(query, k=10)
        print("retrieved docs for router",retrieved_docs)

        for doc in retrieved_docs:
            label = doc.metadata.get("label")
            if label not in seen_labels:
                filtered_docs.append(doc)
                seen_labels.add(label)
            if len(filtered_docs) >= top_k:
                break
        
        if not filtered_docs and retrieved_docs:
            filtered_docs = retrieved_docs[:top_k]
        elif not filtered_docs:
            print("Warning: No semantically similar examples found by router for few-shot prompting.")

        example_text = "\n".join(
            [f'- "{doc.page_content}" → {doc.metadata["label"]}' for doc in filtered_docs]
        )

        routing_prompt_template = PromptTemplate(
            input_variables=["example_text", "query"],
            template="""
You are an intelligent classifier for a multi-agent system. Your job is to route the user's query to one of the following agents or agent combinations:

- car_agent: For queries about car specifications, models, comparisons, etc. (e.g., "horsepower of Honda Civic", "fastest SUV")
- country_agent: For queries about country information, capitals, population, etc. (e.g., "population of India", "capital of Canada")
- math_agent: For direct mathematical calculations. (e.g., "56 * (45 + 2)", "square root of 144")
- code_agent: For queries related to code generation or explanation. (e.g., "write a python function for X")
- car_math_agent: For queries that require fetching car data first AND then performing a mathematical operation on that data. (e.g., "average price of Toyota cars", "sum of NCAP ratings for sedans", "mean of all ncap ratings")
- country_math_agent: For queries that require fetching country data first AND then performing a mathematical operation on that data. (e.g., "total population of listed Scandinavian countries", "average area of European countries in the dataset")
- fallback: If the query doesn't fit any other category, is too ambiguous, or requests unsupported operations like live currency conversion.



Based on your understanding of the categories, classify the following query:
"{query}"

Return only one label from the list: car_agent, country_agent, math_agent, code_agent, car_math_agent, country_math_agent, fallback.
            """
        )

        routing_chain = routing_prompt_template | self.llm | StrOutputParser()
        
        response = routing_chain.invoke({"example_text": example_text, "query": query}).strip().lower()
        print("router response",response)
        valid_responses = {
            "car_agent", 
            "country_agent", 
            "math_agent", 
            "code_agent", 
            "car_math_agent", 
            "country_math_agent",
            "fallback"
        }
        if response not in valid_responses:
            print(f"Router LLM produced an invalid route: '{response}'. Defaulting to fallback.")
            response = "fallback"
        return response

