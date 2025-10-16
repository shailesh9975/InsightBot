# tests/test_insight_bot.py
import os
import shutil
import pytest
from src.insight_bot import InsightBot
from src.data_processor import DataProcessor
from src.config import RAW_DATA_DIR, CHROMA_DB_DIR, OPENAI_API_KEY
from unittest.mock import patch, MagicMock

# Fixture to prepare the vector store for InsightBot tests
@pytest.fixture(scope="module")
def setup_insightbot_vectorstore():
    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY is not set. Skipping InsightBot tests that require OpenAI.")
    
    test_db_dir = os.path.join(CHROMA_DB_DIR, "test_insightbot_db")
    test_raw_dir = os.path.join(RAW_DATA_DIR, "test_insightbot_raw")
    
    # Ensure cleanup first
    if os.path.exists(test_db_dir):
        shutil.rmtree(test_db_dir)
    if os.path.exists(test_raw_dir):
        shutil.rmtree(test_raw_dir)
    
    os.makedirs(test_raw_dir, exist_ok=True)
    
    # Create test documents
    with open(os.path.join(test_raw_dir, "project_details.txt"), "w") as f:
        f.write("InsightBot is a RAG-powered chatbot. It aims to provide accurate answers from user documents. It uses LangChain and OpenAI.")
    with open(os.path.join(test_raw_dir, "tech_stack.txt"), "w") as f:
        f.write("The technology stack includes Python, LangChain, OpenAI's GPT-3.5-turbo, ChromaDB for vector storage, and Streamlit for the UI.")
    
    # Ingest data using DataProcessor
    processor = DataProcessor()
    processor.ingest_data(test_raw_dir, test_db_dir)
    
    yield test_db_dir
    
    # Cleanup
    if os.path.exists(test_db_dir):
        shutil.rmtree(test_db_dir)
    if os.path.exists(test_raw_dir):
        shutil.rmtree(test_raw_dir)


@pytest.fixture(scope="module")
def insight_bot_instance(setup_insightbot_vectorstore):
    # Temporarily override CHROMA_DB_DIR in config for the test
    with patch('src.config.CHROMA_DB_DIR', setup_insightbot_vectorstore):
        bot = InsightBot()
        yield bot


def test_insight_bot_initialization(insight_bot_instance):
    assert insight_bot_instance.vectorstore is not None
    assert insight_bot_instance.retriever is not None
    assert insight_bot_instance.llm is not None
    assert insight_bot_instance.qa_chain is not None


def test_insight_bot_initialization_no_vectorstore():
    # Test behavior when no vector store is found
    non_existent_db_dir = os.path.join(CHROMA_DB_DIR, "non_existent_temp_db")
    if os.path.exists(non_existent_db_dir):
        shutil.rmtree(non_existent_db_dir) # Ensure it doesn't exist

    with patch('src.config.CHROMA_DB_DIR', non_existent_db_dir):
        with pytest.raises(RuntimeError, match="Vector store not initialized"):
            InsightBot()

def test_ask_question_valid(insight_bot_instance):
    # This test will make an actual API call to OpenAI
    # Ensure OPENAI_API_KEY is properly set in your .env
    query = "What is InsightBot?"
    response = insight_bot_instance.ask(query)
    
    assert "answer" in response
    assert "sources" in response
    assert isinstance(response["answer"], str)
    assert isinstance(response["sources"], list)
    assert len(response["answer"]) > 0
    # The answer should ideally contain keywords from the context
    assert "RAG-powered chatbot" in response["answer"] or "accurate answers from user documents" in response["answer"] or "LangChain" in response["answer"]

    assert len(response["sources"]) > 0
    assert any("project_details.txt" in s['metadata'].get('source', '') for s in response["sources"])


def test_ask_question_no_relevant_info(insight_bot_instance):
    # This query should not find direct answers in our small dataset
    query = "What is the capital of France?"
    response = insight_bot_instance.ask(query)
    
    assert "answer" in response
    assert "sources" in response
    assert isinstance(response["answer"], str)
    # The LLM should ideally respond that it doesn't know based on the prompt
    assert "don't know" in response["answer"].lower() or "couldn't find" in response["answer"].lower()
    
    # Sources might still be returned if embeddings are somewhat similar, but the answer should reflect lack of context.
    # The important part is that the answer indicates no relevant info.
    
def test_ask_question_empty_query(insight_bot_instance):
    response = insight_bot_instance.ask("")
    assert response["answer"] == "Please provide a question."
    assert response["sources"] == []

# Mock the OpenAI API call for faster, non-API-dependent testing
@patch('langchain_openai.ChatOpenAI.invoke')
@patch('langchain_community.vectorstores.Chroma.as_retriever')
def test_ask_question_mocked_llm(mock_as_retriever, mock_llm_invoke, setup_insightbot_vectorstore):
    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY is not set. Skipping tests that require OpenAIEmbeddings, even when mocking LLM.")

    # Configure the mock retriever to return specific documents
    mock_retriever_instance = MagicMock()
    mock_retriever_instance.invoke.return_value = [
        Document(page_content="Mocked context about InsightBot.", metadata={"source": "mock_doc.txt"})
    ]
    mock_as_retriever.return_value = mock_retriever_instance

    # Configure the mock LLM to return a specific answer
    mock_llm_invoke.return_value.content = "This is a mocked answer about InsightBot."

    with patch('src.config.CHROMA_DB_DIR', setup_insightbot_vectorstore):
        bot = InsightBot()
        query = "Tell me about InsightBot."
        response = bot.ask(query)
        
        assert response["answer"] == "This is a mocked answer about InsightBot."
        assert len(response["sources"]) == 1
        assert response["sources"][0]["metadata"]["source"] == "mock_doc.txt"
        
        # Verify that the retriever and LLM were called as expected
        mock_retriever_instance.invoke.assert_called_once_with(query)
        # Note: The actual prompt passed to LLM invoke is complex, so we check if it was called.
        mock_llm_invoke.assert_called_once()
