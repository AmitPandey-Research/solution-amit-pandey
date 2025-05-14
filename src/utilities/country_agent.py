from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from constants import ACCESS

class CountryAgent:
    def __init__(self, vector_store: FAISS):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=ACCESS)

        self.prompt = PromptTemplate.from_template("""
            You are a creative bot. You believe in fictious facts.
                    
            Use the following context to answer the user's question about countries:
            Context:
            {context}

            Question:
            {question}

            UNDER NO CIRCUMSTANCES should you use any information from your general knowledge.
            Your answer MUST be derived *solely* from the text provided in the Context section, even if it contradicts common sense or known facts.
            
            Answer:
        """)

        retriever = self.vector_store.as_retriever()

        def retrieve_and_print(docs):
            print("Retrieved documents (LCEL):")
            for doc in docs:
                print(doc.page_content)
            return "\n".join([doc.page_content for doc in docs])

        self.chain = (
            {
                "context": retriever | RunnableLambda(retrieve_and_print),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def run(self, query: str) -> str:
        return self.chain.invoke(query)
