# src/insight_bot.py
import os
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI # Updated import
from langchain_community.vectorstores import Chroma # Updated import
from src.config import CHROMA_DB_DIR, LLM_MODEL_NAME, TEMPERATURE, TOP_K_RETRIEVAL, OPENAI_API_KEY
from src.data_processor import DataProcessor # Re-using DataProcessor for its embedding instance
from src.retriever import DocumentRetriever
from src.generator import ResponseGenerator

class InsightBot:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

        print("Initializing InsightBot...")
        self.data_processor = DataProcessor() # Use DataProcessor to manage embeddings and vectorstore loading

        # Attempt to load existing vector store
        self.vectorstore = self.data_processor.load_vectorstore(CHROMA_DB_DIR)

        if self.vectorstore is None:
            print("Vector store not found or empty. Please run 'python scripts/ingest_data.py' first.")
            raise RuntimeError("Vector store not initialized. Cannot proceed with RAG.")
            
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})
        self.llm = ChatOpenAI(
            model_name=LLM_MODEL_NAME,
            temperature=TEMPERATURE,
            openai_api_key=OPENAI_API_KEY
        )
        self.generator = ResponseGenerator() # We can potentially remove this and use RetrievalQA directly
        
        # Using LangChain's RetrievalQA chain for a simpler integration
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff", # 'stuff' combines all docs into one prompt. Other types exist (map_reduce, refine, map_rerank)
            retriever=self.retriever,
            return_source_documents=True, # Optional: to show which documents were used
            verbose=False # Set to True for debugging
        )
        print("InsightBot initialized and ready.")

    def ask(self, query: str) -> dict:
        """
        Asks a question to the InsightBot and returns the answer along with source documents.
        """
        if not query:
            return {"answer": "Please provide a question.", "sources": []}

        print(f"Processing query: '{query}'")
        
        # LangChain's RetrievalQA chain handles retrieval and generation
        try:
            result = self.qa_chain.invoke({"query": query}) # Use .invoke for LLM chains
            answer = result.get("result", "Sorry, I couldn't find an answer based on the provided documents.")
            source_documents = result.get("source_documents", [])
            sources = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in source_documents]
            
            print("Query processed successfully.")
            return {"answer": answer, "sources": sources}
        except Exception as e:
            print(f"An error occurred during query processing: {e}")
            return {"answer": f"An error occurred: {e}", "sources": []}
