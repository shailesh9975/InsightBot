# src/data_processor.py
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings # Updated import for OpenAIEmbeddings
from langchain_community.vectorstores import Chroma # Updated import for Chroma
from langchain.schema import Document
from src.config import RAW_DATA_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, OPENAI_API_KEY

class DataProcessor:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
        
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, openai_api_key=OPENAI_API_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""], # More robust splitting
            length_function=len,
            is_separator_regex=False,
        )
        self.vectorstore = None # Will be initialized or loaded later

    def load_documents_from_directory(self, directory_path: str = RAW_DATA_DIR) -> List[Document]:
        """Loads documents from a specified directory, supporting PDF and TXT files."""
        print(f"Loading documents from: {directory_path}")
        documents = []
        
        # Load PDF files
        pdf_loader = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader, silent_errors=True)
        documents.extend(pdf_loader.load())
        print(f"Loaded {len(documents)} PDF documents.")

        # Load TXT files
        txt_loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True)
        documents.extend(txt_loader.load())
        print(f"Loaded {len(documents) - len(pdf_loader.load())} TXT documents.")
        
        # Add more loaders as needed (e.g., MarkdownLoader)

        if not documents:
            print(f"No documents found in {directory_path}.")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits loaded documents into smaller chunks."""
        print(f"Splitting {len(documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks.")
        return chunks

    def create_and_save_vectorstore(self, chunks: List[Document], db_path: str = CHROMA_DB_DIR):
        """Creates a Chroma vector store from document chunks and saves it to disk."""
        print(f"Creating vector store at: {db_path}")
        # Ensure the directory exists
        os.makedirs(db_path, exist_ok=True)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=db_path
        )
        self.vectorstore.persist()
        print("Vector store created and saved successfully.")

    def load_vectorstore(self, db_path: str = CHROMA_DB_DIR):
        """Loads an existing Chroma vector store from disk."""
        print(f"Loading vector store from: {db_path}")
        if not os.path.exists(db_path):
            print(f"Vector store directory not found at {db_path}. It will be created on ingestion.")
            self.vectorstore = None # Ensure it's None if not found
            return None
        
        self.vectorstore = Chroma(
            persist_directory=db_path, 
            embedding_function=self.embeddings
        )
        print("Vector store loaded successfully.")
        return self.vectorstore

    def ingest_data(self, directory_path: str = RAW_DATA_DIR, db_path: str = CHROMA_DB_DIR):
        """Full pipeline to load, split, embed, and save documents to a vector store."""
        documents = self.load_documents_from_directory(directory_path)
        if not documents:
            print("No documents to ingest. Please add files to the data/raw directory.")
            return

        chunks = self.split_documents(documents)
        self.create_and_save_vectorstore(chunks, db_path)
        print("Data ingestion complete.")
