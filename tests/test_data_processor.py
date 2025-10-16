import os
import shutil
import pytest
from unittest.mock import patch
from src.data_processor import DataProcessor

@pytest.fixture
def data_processor_instance():
    return DataProcessor()

@pytest.fixture
def temp_data_setup(tmp_path):
    # Create temporary directories for raw data and vector store
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    db_dir = tmp_path / "chroma_db"
    db_dir.mkdir()
    # Create a test TXT document
    (raw_dir / "test_doc.txt").write_text(
        "This is a test document. It contains some text for processing."
    )
    return str(raw_dir), str(db_dir)


def test_data_processor_initialization(data_processor_instance):
    assert data_processor_instance is not None
    assert hasattr(data_processor_instance, "embeddings")


def test_split_documents(data_processor_instance, temp_data_setup):
    test_raw_dir, _ = temp_data_setup
    documents = data_processor_instance.load_documents_from_directory(test_raw_dir)
    if not documents:
        pytest.skip("No documents loaded to split.")
    chunks = data_processor_instance.split_documents(documents)
    assert len(chunks) > 0  # At least 1 chunk
    assert all(hasattr(c, "page_content") for c in chunks)


@patch("src.data_processor.OpenAIEmbeddings.embed_documents")
def test_create_and_save_vectorstore(mock_embed, data_processor_instance, temp_data_setup):
    test_raw_dir, test_db_dir = temp_data_setup
    documents = data_processor_instance.load_documents_from_directory(test_raw_dir)
    chunks = data_processor_instance.split_documents(documents)

    # Return the same number of fake embeddings as chunks
    mock_embed.return_value = [[0.0] * 1536 for _ in chunks]

    data_processor_instance.create_and_save_vectorstore(chunks, test_db_dir)
    # Ensure vectorstore is created
    assert data_processor_instance.vectorstore is not None


@patch("src.data_processor.OpenAIEmbeddings.embed_documents")
def test_load_vectorstore(mock_embed, data_processor_instance, temp_data_setup):
    test_raw_dir, test_db_dir = temp_data_setup
    documents = data_processor_instance.load_documents_from_directory(test_raw_dir)
    chunks = data_processor_instance.split_documents(documents)

    # Mock embeddings for all chunks
    mock_embed.return_value = [[0.0] * 1536 for _ in chunks]

    # Create and save first
    data_processor_instance.create_and_save_vectorstore(chunks, test_db_dir)

    # Now reload vectorstore
    loaded_vs = data_processor_instance.load_vectorstore(test_db_dir)
    assert loaded_vs is not None


@patch("src.data_processor.OpenAIEmbeddings.embed_documents")
def test_ingest_data_pipeline(mock_embed, data_processor_instance, temp_data_setup):
    test_raw_dir, test_db_dir = temp_data_setup
    documents = data_processor_instance.load_documents_from_directory(test_raw_dir)
    chunks = data_processor_instance.split_documents(documents)

    # Mock embeddings for all chunks
    mock_embed.return_value = [[0.0] * 1536 for _ in chunks]

    # Ensure previous db_dir is clean
    if os.path.exists(test_db_dir):
        shutil.rmtree(test_db_dir)

    data_processor_instance.ingest_data(test_raw_dir, test_db_dir)
    assert os.path.exists(test_db_dir)
    assert data_processor_instance.vectorstore is not None


def test_missing_openai_api_key_raises_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from src.data_processor import DataProcessor

    with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
        DataProcessor()

