# src/generator.py
from langchain_community.chat_models import ChatOpenAI # Updated import for ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config import LLM_MODEL_NAME, TEMPERATURE, OPENAI_API_KEY

class ResponseGenerator:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
            
        self.llm = ChatOpenAI(
            model_name=LLM_MODEL_NAME,
            temperature=TEMPERATURE,
            openai_api_key=OPENAI_API_KEY
        )
        self.prompt_template = ChatPromptTemplate.from_template(
            """
            You are an AI assistant for the InsightBot project. Use the following context to answer the question.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Keep the answer concise and to the point, referring only to the provided context.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
        )
        print(f"ResponseGenerator initialized with LLM: {LLM_MODEL_NAME}, Temperature: {TEMPERATURE}")

    def generate_response(self, question: str, context: str) -> str:
        """
        Generates a response using the LLM based on the question and provided context.
        """
        if not question:
            return "Please provide a question."
        if not context:
            print("Warning: No context provided for generation. LLM might hallucinate or respond generally.")
            # We can modify the prompt or add a fallback here if no context is available
            return self.llm.invoke(f"Answer the following question without any specific context: {question}").content

        # Using LangChain Expression Language (LCEL) for a chain
        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()} # Context will come from retriever/chain
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        # When calling with .invoke, the context is the *result* of the retriever, which is a list of docs.
        # However, generate_response directly expects a string context. We will adapt this when building InsightBot.
        # For now, let's ensure this method can be called directly.
        # A simple invoke for testing purposes:
        response = self.llm.invoke(self.prompt_template.format(context=context, question=question)).content
        print(f"Generated response for question: '{question}'")
        return response
