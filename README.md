code
Markdown
# InsightBot: Your Personalized Knowledge Assistant

InsightBot is a Retrieval-Augmented Generation (RAG) powered chatbot designed to answer questions accurately and contextually from a user-provided set of documents. This project focuses on building a robust, maintainable, and deployable RAG system from the ground up.

## Table of Contents
	- [Project Goal](#project-goal)
	- [Features](#features)
	- [Architecture](#architecture)
	- [Setup & Installation](#setup--installation)
	- [Usage](#usage)
	- [Development Log](#development-log)
	- [Testing](#testing)
	- [Deployment](#deployment)
	- [Contributing](#contributing)
	- [License](#license)

## Project Goal
To create a RAG-powered chatbot that can answer questions accurately and contextually from a user-provided set of documents, focusing on maintainability, testability, and deployability.

## Features
	- **Document Ingestion:** Load various document types (PDF, TXT, MD) from a specified directory.
	- **Intelligent Chunking:** Split documents into optimal-sized chunks for efficient retrieval.
	- **Vector Store Integration:** Utilize ChromaDB for storing and querying document embeddings.
	- **Contextual Retrieval:** Retrieve the most relevant document chunks based on user queries.
	- **LLM-Powered Generation:** Leverage a Large Language Model (e.g., OpenAI GPT-3.5) to generate coherent and accurate answers.
	- **Streamlit UI:** An easy-to-use web interface for interacting with the chatbot.
	- **Robust Testing:** Comprehensive unit and integration tests for core components.
	- **Environment Management:** Secure handling of API keys and configurations using `.env`.

## Architecture
```mermaid
graph TD
    A[User Query] --> B(Streamlit UI)
    B --> C(InsightBot Application)
    C -- "Query" --> D[Retriever]
    D -- "Context" --> E[Generator]
    E -- "Answer" --> C
    C --> B
    D -- "Embed Query" --> F(Embedding Model)
    F -- "Vector Search" --> G[Vector Database (ChromaDB)]
    G -- "Relevant Chunks" --> D
    H[Data Ingestion (scripts/ingest_data.py)] --> I(Data Processor)
    I -- "Embed Chunks" --> F
    F -- "Store Embeddings" --> G
    J[Raw Documents (data/raw/)] --> H
Setup & Installation
Prerequisites
Python 3.9+
Git
Steps
Clone the repository:
code
Bash
git clone https://github.com/yourusername/insightbot.git # Replace with your repo URL
cd insightbot
Create and activate a virtual environment:
code
Bash
python -m venv .venv
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
Install dependencies:
code
Bash
pip install -r requirements.txt
Set up Environment Variables:
Create a .env file in the root directory of the project and add your OpenAI API key:
code
Code
OPENAI_API_KEY="your_openai_api_key_here"
Ensure this file is not committed to Git (it's in .gitignore).
Usage
(To be filled in as we build the project)

##Development Log

Day 0: Project Setup & Foundation
Initialized Git repository.
Created .gitignore.
Set up Python virtual environment.
Installed initial core dependencies (langchain, openai, chromadb, pypdf, python-dotenv, streamlit, pytest).
Generated requirements.txt.
Established core directory structure (src, data, tests, scripts).
Created placeholder files and initial README.md.
Outlined project goal and initial architecture.
Testing
(To be filled in as we write tests)


##Deployment

## Development Log

### Day 0: Project Setup & Foundation
- Initialized Git repository.
- Created `.gitignore`.
- Set up Python virtual environment.
- Installed initial core dependencies (`langchain`, `openai`, `chromadb`, `pypdf`, `python-dotenv`, `streamlit`, `pytest`).
- Generated `requirements.txt`.
- Established core directory structure (`src`, `data`, `tests`, `scripts`).
- Created placeholder files and initial `README.md`.
- Outlined project goal and initial architecture.

### Day 1: Configuration & Data Processing Core
- Implemented `src/config.py` for centralized project settings and API key management.
- Developed `src/data_processor.py` for:
    - Loading documents (PDF, TXT) using `DirectoryLoader`.
    - Splitting documents into chunks using `RecursiveCharacterTextSplitter`.
    - Creating and saving a `Chroma` vector store with `OpenAIEmbeddings`.
    - Loading an existing `Chroma` vector store.
- Created `scripts/ingest_data.py` to run the data processing pipeline from the command line.
- Added sample documents to `data/raw/`.
- Wrote comprehensive unit tests (`tests/test_data_processor.py`) for `DataProcessor` functionalities, including loading, splitting, vector store creation/loading, and API key handling.
- Verified local ingestion and all tests passed.

### Day 2: Retrieval & Generation Core
- Implemented `src/retriever.py` with `DocumentRetriever` class to:
    - Initialize with a Chroma vector store.
    - Retrieve top-k relevant document chunks based on a query using `vectorstore.as_retriever()`.
    - Concatenate retrieved document content into a single context string.
- Implemented `src/generator.py` with `ResponseGenerator` class (though its direct usage is simplified by `RetrievalQA`):
    - Initialize `ChatOpenAI` with configurable model and temperature.
    - Defined a robust chat prompt template for RAG.
    - `generate_response` method for LLM interaction (primarily for testing direct calls).
- Implemented `src/insight_bot.py` with `InsightBot` class:
    - Orchestrates the RAG flow: loads vector store, sets up retriever and LLM.
    - Uses LangChain's `RetrievalQA.from_chain_type` for streamlined RAG processing.
    - `ask` method to take a query and return an answer with source documents.
- Wrote unit and integration tests for `DocumentRetriever` (`tests/test_retriever.py`) and `InsightBot` (`tests/test_insight_bot.py`), including mocked LLM calls for efficiency and real API calls for full integration.
- Verified all new tests passed, confirming retrieval and generation logic.



##License
(e.g., MIT License)
code
Code
**Step 0.3.4: Final Git Commit for Day 0**

```bash
git add .
git commit -m "feat: Initial project structure and README.md"
