# db_utils.py

import sqlite3
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_db():
    """
    Initialize SQLite database.
    Creates the table if it doesn't exist and adds the 'page_count' column
    if it's missing from an older version of the database.
    """
    conn = None
    try:
        conn = sqlite3.connect("nalco_chatbot.db")
        c = conn.cursor()
        
        # Create the table if it doesn't exist
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                file_name TEXT PRIMARY KEY,
                extracted_text TEXT,
                timestamp TEXT
            )
            """
        )
        
        # --- START OF FIX: Schema Migration ---
        # Check if the 'page_count' column exists
        c.execute("PRAGMA table_info(documents)")
        columns = [info[1] for info in c.fetchall()]
        
        if 'page_count' not in columns:
            logger.warning("Old database schema detected. Adding 'page_count' column.")
            # Add the column if it doesn't exist
            c.execute("ALTER TABLE documents ADD COLUMN page_count INTEGER DEFAULT 0")
        # --- END OF FIX ---

        conn.commit()
        logger.info("SQLite database initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Database initialization failed: {e}")
        raise # Re-raise the exception to stop the app if DB fails
    finally:
        if conn:
            conn.close()

def store_document(file_name: str, extracted_text: str, timestamp: str = None, page_count: int = None):
    """Store the extracted text, metadata, and timestamp in SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect("nalco_chatbot.db")
        c = conn.cursor()
        
        # Using a default of 0 for page_count if None is provided
        page_count_to_store = page_count if page_count is not None else 0
        
        c.execute(
            "INSERT OR REPLACE INTO documents (file_name, extracted_text, timestamp, page_count) VALUES (?, ?, ?, ?)",
            (file_name, extracted_text, timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"), page_count_to_store)
        )
        conn.commit()
        logger.info(f"Stored document in SQLite: {file_name}")
    except sqlite3.Error as e:
        logger.error(f"Failed to store document {file_name}: {e}")
    finally:
        if conn:
            conn.close()

def load_documents_from_db():
    """Load all documents, their extracted texts, and timestamps from SQLite database."""
    conn = None
    documents = []
    try:
        conn = sqlite3.connect("nalco_chatbot.db")
        c = conn.cursor()
        # This query will now succeed because init_db ensures the column exists
        c.execute("SELECT file_name, extracted_text, timestamp, page_count FROM documents")
        documents = c.fetchall()
        logger.info(f"Loaded {len(documents)} documents from SQLite.")
    except sqlite3.Error as e:
        logger.error(f"Failed to load documents from database: {e}")
    finally:
        if conn:
            conn.close()
    return documents

if __name__ == "__main__":
    init_db()
