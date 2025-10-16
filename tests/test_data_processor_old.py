# tests/test_data_processor.py
import os
import shutil
import pytest
from src.data_processor import DataProcessor
from src.config import RAW_DATA_DIR, CHROMA_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, OPENAI_API_KEY

# Fixture to create temporary test data directory and cleanup
@pytest.fixture(scope="module")
def temp_data_setup():
    test_raw_dir = os.path.join(RAW_DATA_DIR, "test_raw")
    test_db_dir = os.path.join(CHROMA_DB_DIR, "test_db")
    
    os.makedirs(test_raw_dir, exist_ok=True)
    
    # Create dummy text file
    with open(os.path.join(test_raw_dir, "test_doc.txt"), "w") as f:
        f.write("This is a test document. It contains some text for processing. " * 10) # Enough text to chunk
    
    # Create dummy PDF file (we can't easily create a valid PDF here, so we'll just check if it's ignored or loaded)
    # For actual PDF loading, you'd need a real PDF file. For this test, we assume PyPDFLoader handles its input.
    # We'll mostly test the text loading and splitting for simplicity.
    
    yield test_raw_dir, test_db_dir
    
    # Cleanup
    if os.path.exists(test_raw_dir):
        shutil.rmtree(test_raw_dir)
    if os.path.exists(test_db_dir):
        shutil.rmtree(test_db_dir)

@pytest.fixture(scope="module")
def data_processor_instance():
    # Ensure OPENAI_API_KEY is set for tests
    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY is not set. Skipping tests that require OpenAIEmbeddings.")
    return DataProcessor()


def test_data_processor_initialization(data_processor_instance):
    assert data_processor_instance.embeddings is not None
    assert data_processor_instance.text_splitter is not None
    assert data_processor_instance.vectorstore is None


def test_load_documents_from_directory(data_processor_instance, temp_data_setup):
    test_raw_dir, _ = temp_data_setup
    documents = data_processor_instance.load_documents_from_directory(test_raw_dir)
    assert len(documents) >= 1 # Should load at least the test_doc.txt
    assert any("test_doc.txt" in doc.metadata.get('source', '') for doc in documents)


def test_split_documents(data_processor_instance, temp_data_setup):
    test_raw_dir, _ = temp_data_setup
    documents = data_processor_instance.load_documents_from_directory(test_raw_dir)
    if not documents:
        pytest.skip("No documents loaded to split.")
    chunks = data_processor_instance.split_documents(documents)
    assert len(chunks) > 1 # Expecting multiple chunks from a long document
    assert all(len(chunk.page_content) <= CHUNK_SIZE for chunk in chunks)


def test_create_and_save_vectorstore(data_processor_instance, temp_data_setup):
    _, test_db_dir = temp_data_setup
    test_raw_dir, _ = temp_data_setup
    
    documents = data_processor_instance.load_documents_from_directory(test_raw_dir)
    if not documents:
        pytest.skip("No documents loaded to create vectorstore.")
        
    chunks = data_processor_instance.split_documents(documents)
    
    data_processor_instance.create_and_save_vectorstore(chunks, test_db_dir)
    assert data_processor_instance.vectorstore is not None
    assert os.path.exists(test_db_dir) # Check if directory was created
    assert len(os.listdir(test_db_dir)) > 0 # Check if files were written


def test_load_vectorstore(data_processor_instance, temp_data_setup):
    _, test_db_dir = temp_data_setup
    # First, ensure a vectorstore is created and saved
    test_create_and_save_vectorstore(data_processor_instance, temp_data_setup)
    
    # Now try to load it
    loaded_vectorstore = data_processor_instance.load_vectorstore(test_db_dir)
    assert loaded_vectorstore is not None
    assert isinstance(loaded_vectorstore, Chroma)
    
    # Test loading a non-existent vectorstore
    non_existent_db_dir = os.path.join(CHROMA_DB_DIR, "non_existent")
    if os.path.exists(non_existent_db_dir):
        shutil.rmtree(non_existent_db_dir)
    
    non_existent_processor = DataProcessor() # New instance to avoid state
    loaded_none = non_existent_processor.load_vectorstore(non_existent_db_dir)
    assert loaded_none is None # Should return None if not found
    assert non_existent_processor.vectorstore is None


def test_ingest_data_pipeline(data_processor_instance, temp_data_setup):
    test_raw_dir, test_db_dir = temp_data_setup
    
    # Ensure previous test_db_dir is clean for this pipeline test
    if os.path.exists(test_db_dir):
        shutil.rmtree(test_db_dir)
        
    data_processor_instance.ingest_data(test_raw_dir, test_db_dir)
    assert data_processor_instance.vectorstore is not None
    assert os.path.exists(test_db_dir)
    assert len(os.listdir(test_db_dir)) > 0

# Test for missing OPENAI_API_KEY
def test_missing_openai_api_key_raises_error():
    # Temporarily unset the API key to test this scenario
    original_key = os.environ.pop("OPENAI_API_KEY", None)
    
    try:
        with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
            DataProcessor()
    finally:
        # Restore the API key
        if original_key is not None:
            os.environ["OPENAI_API_KEY"] = original_key
