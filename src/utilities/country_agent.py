from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from constants import ACCESS

class CountryAgent:
    def __init__(self, vector_store: FAISS):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=ACCESS)
        self.prompt = PromptTemplate.from_template("""
            Use the following context to answer the user's question about countries:
            Context:
            {context}

            Question:
            {question}

            Answer:
            """)

        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            return_source_documents=False,
        )

    def refine_query(self, query: str, context: str) -> str:
        """Refines the query based on the context."""
        refine_prompt_template = """
            Given the previous conversation and the current question, refine the current question to be a standalone question.

            Previous Conversation:
            {context}

            Current Question:
            {question}

            Standalone Question:
            """
        refine_prompt = PromptTemplate.from_template(refine_prompt_template)
        refine_chain = refine_prompt | self.llm | StrOutputParser()
        return refine_chain.invoke({"question": query, "context": context})

    def run(self, query: str, context: str = "") -> str:
        """Runs the Country Agent to answer a query."""
        refined_query = self.refine_query(query, context)
        answer = self.chain.invoke({"question": refined_query, "context": self.get_relevant_documents(refined_query)})
        return answer["result"]

    def get_relevant_documents(self, query: str) -> str:
        docs = self.vector_store.similarity_search(query)
        return "\n".join([doc.page_content for doc in docs])