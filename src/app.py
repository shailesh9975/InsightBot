# src/app.py
import streamlit as st
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.insight_bot import InsightBot
from src.config import APP_TITLE, RAW_DATA_DIR, CHROMA_DB_DIR
from scripts.ingest_data import main as ingest_data_script # Import the main function from ingest_data.py
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize InsightBot
# We'll use Streamlit's st.session_state to manage the bot instance
# and st.cache_resource to ensure it's loaded only once across reruns.
@st.cache_resource
def load_insight_bot():
    """Loads and caches the InsightBot instance."""
    try:
        logger.info("Attempting to load InsightBot...")
        bot = InsightBot()
        logger.info("InsightBot loaded successfully.")
        return bot
    except RuntimeError as e:
        logger.error(f"Failed to initialize InsightBot: {e}")
        st.error(f"Error initializing InsightBot: {e}. "
                 "Please ensure documents are ingested by running 'python scripts/ingest_data.py'.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during InsightBot loading: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {e}. Check server logs for details.")
        return None

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🤖")
    st.title(APP_TITLE)
    st.write("Ask questions about your documents!")

    # Check for API key presence early
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("`OPENAI_API_KEY` not found in environment variables. Please set it in your `.env` file.")
        st.stop() # Stop the app if API key is missing

    # Load InsightBot
    bot = load_insight_bot()
    if bot is None:
        st.info("No documents found in the knowledge base. Please upload documents and run ingestion.")
        # Provide an option to run ingestion from UI for convenience (requires admin-like access or specific setup)
        if st.button("Re-run Data Ingestion"):
            with st.spinner("Ingesting data... This may take a moment."):
                try:
                    # Call the main ingestion function
                    ingest_data_script(directory_path=RAW_DATA_DIR, db_path=CHROMA_DB_DIR)
                    st.success("Data ingestion complete! Please refresh the page to load the updated knowledge base.")
                    st.cache_resource.clear() # Clear cache to force reload
                except Exception as e:
                    st.error(f"Error during data ingestion: {e}")
                    logger.error(f"Error during ingestion: {e}", exc_info=True)
            st.stop() # Stop here, user needs to refresh for bot to load

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            try:
                response_data = bot.ask(prompt)
                full_response = response_data["answer"]
                sources = response_data["sources"]

                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(full_response)
                    if sources:
                        st.expander("Sources").json(sources) # Show source metadata
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": sources})
            except Exception as e:
                logger.error(f"Error during chatbot interaction: {e}", exc_info=True)
                st.chat_message("assistant").error(f"An error occurred while processing your request: {e}. Please try again.")
                st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {e}"})

if __name__ == "__main__":
    main()
