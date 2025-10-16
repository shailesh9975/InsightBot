# scripts/ingest_data.py (Updated)
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processor import DataProcessor
from src.config import RAW_DATA_DIR, CHROMA_DB_DIR
import logging

logger = logging.getLogger(__name__)

def main(directory_path: str = RAW_DATA_DIR, db_path: str = CHROMA_DB_DIR):
    """
    Main function for data ingestion, can be called from script or imported.
    """
    logger.info("Starting data ingestion process for InsightBot...")
    processor = DataProcessor()
    
    os.makedirs(directory_path, exist_ok=True)
    
    if not any(os.listdir(directory_path)):
        logger.warning(f"The '{directory_path}' directory is empty. Please add your documents (e.g., PDFs, TXT files) here.")
        # Optionally, raise an error or return a status indicating no data
        return False # Indicate no data was ingested
    else:
        processor.ingest_data(directory_path=directory_path, db_path=db_path)
        logger.info("Data ingestion process finished.")
        return True # Indicate successful ingestion

if __name__ == "__main__":
    # When run as a script, execute the main function
    main()
