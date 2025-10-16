# src/retriever.py
from typing import List
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from src.config import TOP_K_RETRIEVAL

class DocumentRetriever:
    def __init__(self, vectorstore: Chroma):
        if not vectorstore:
            raise ValueError("Vectorstore must be provided for DocumentRetriever.")
        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})
        print(f"DocumentRetriever initialized with top_k={TOP_K_RETRIEVAL}")

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieves relevant document chunks based on the query.
        """
        if not query:
            return []
        print(f"Retrieving documents for query: '{query}'")
        retrieved_docs = self.retriever.invoke(query)
        print(f"Retrieved {len(retrieved_docs)} relevant document chunks.")
        return retrieved_docs

    def get_context_from_documents(self, documents: List[Document]) -> str:
        """
        Concatenates the content of retrieved documents into a single string to serve as context for the LLM.
        """
        if not documents:
            return ""
        context = "\n\n".join([doc.page_content for doc in documents])
        return context
