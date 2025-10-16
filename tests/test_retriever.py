# tests/test_retriever.py
import os
import shutil
import pytest
from src.data_processor import DataProcessor
from src.retriever import DocumentRetriever
from src.config import RAW_DATA_DIR, CHROMA_DB_DIR, OPENAI_API_KEY, TOP_K_RETRIEVAL
from langchain.schema import Document

# Fixture to set up a temporary ChromaDB with some test data
@pytest.fixture(scope="module")
def setup_chromadb_for_retriever():
    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY is not set. Skipping tests that require OpenAIEmbeddings.")

    temp_db_dir = os.path.join(CHROMA_DB_DIR, "test_retriever_db")
    if os.path.exists(temp_db_dir):
        shutil.rmtree(temp_db_dir)
    
    processor = DataProcessor()
    
    # Create some dummy chunks
    chunks = [
        Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"source": "doc1.txt"}),
        Document(page_content="Dogs are often considered loyal companions.", metadata={"source": "doc2.txt"}),
        Document(page_content="A cunning fox might outsmart a sleepy canine.", metadata={"source": "doc3.txt"}),
        Document(page_content="Cats are independent and enjoy chasing mice.", metadata={"source": "doc4.txt"}),
        Document(page_content="A very very long sentence that will definitely be split into multiple chunks if the chunk size is small enough to demonstrate chunking, but for this test, we want distinct chunks.", metadata={"source": "doc5.txt"}),
    ]
    
    processor.create_and_save_vectorstore(chunks, temp_db_dir)
    
    # Load the created vectorstore
    vectorstore = processor.load_vectorstore(temp_db_dir)
    
    yield vectorstore
    
    # Cleanup
    if os.path.exists(temp_db_dir):
        shutil.rmtree(temp_db_dir)

def test_document_retriever_initialization(setup_chromadb_for_retriever):
    retriever = DocumentRetriever(setup_chromadb_for_retriever)
    assert retriever.vectorstore is not None
    assert retriever.retriever is not None


def test_get_relevant_documents(setup_chromadb_for_retriever):
    retriever = DocumentRetriever(setup_chromadb_for_retriever)
    
    # Test a query that should match "fox" and "dog" documents
    query = "What is known about a fox and a dog?"
    relevant_docs = retriever.get_relevant_documents(query)
    
    assert len(relevant_docs) <= TOP_K_RETRIEVAL # Should return up to TOP_K_RETRIEVAL documents
    assert len(relevant_docs) > 0 # Should find some documents
    
    # Check if expected content is in the retrieved docs (fuzzy check)
    found_fox_doc = any("fox" in doc.page_content.lower() for doc in relevant_docs)
    found_dog_doc = any("dog" in doc.page_content.lower() for doc in relevant_docs)
    assert found_fox_doc
    assert found_dog_doc

def test_get_relevant_documents_no_match(setup_chromadb_for_retriever):
    retriever = DocumentRetriever(setup_chromadb_for_retriever)
    query = "Completely unrelated term like spaceship technology?"
    relevant_docs = retriever.get_relevant_documents(query)
    
    # It might still return some docs if embeddings are somewhat close, but
    # the content won't be directly relevant. The key is that it doesn't break.
    assert len(relevant_docs) <= TOP_K_RETRIEVAL


def test_get_relevant_documents_empty_query(setup_chromadb_for_retriever):
    retriever = DocumentRetriever(setup_chromadb_for_retriever)
    relevant_docs = retriever.get_relevant_documents("")
    assert len(relevant_docs) == 0


def test_get_context_from_documents():
    retriever = DocumentRetriever(vectorstore=None) # We only need the method, vectorstore isn't used
    docs = [
        Document(page_content="First sentence."),
        Document(page_content="Second sentence."),
        Document(page_content="Third sentence.")
    ]
    context = retriever.get_context_from_documents(docs)
    assert context == "First sentence.\n\nSecond sentence.\n\nThird sentence."

def test_get_context_from_empty_documents():
    retriever = DocumentRetriever(vectorstore=None)
    context = retriever.get_context_from_documents([])
    assert context == ""
