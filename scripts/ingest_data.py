# scripts/ingest_data.py
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processor import DataProcessor
from src.config import RAW_DATA_DIR, CHROMA_DB_DIR

if __name__ == "__main__":
    print("Starting data ingestion process for InsightBot...")
    processor = DataProcessor()
    
    # Ensure the raw data directory exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Check if there are any files in the raw data directory
    if not any(os.listdir(RAW_DATA_DIR)):
        print(f"The '{RAW_DATA_DIR}' directory is empty. Please add your documents (e.g., PDFs, TXT files) here before running ingestion.")
        print("Example: Create a file 'data/raw/example.txt' with some text content.")
    else:
        processor.ingest_data(directory_path=RAW_DATA_DIR, db_path=CHROMA_DB_DIR)
    print("Data ingestion process finished.")
