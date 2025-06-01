import sqlite3
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    """Initialize SQLite database to store extracted texts and metadata."""
    try:
        conn = sqlite3.connect("nalco_chatbot.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                extracted_text TEXT NOT NULL,
                chroma_path TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()
        logger.info("Initialized SQLite database: nalco_chatbot.db")
    except Exception as e:
        logger.error(f"Failed to initialize SQLite database: {e}")
        raise  # Re-raise the exception to alert the app
    finally:
        conn.close()

def store_document(file_name, extracted_text, vector_db_path="./PDF_ChromaDB"):
    """Store extracted text, Chroma DB path, and timestamp in the database."""
    try:
        conn = sqlite3.connect("nalco_chatbot.db")
        cursor = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(
            "INSERT INTO documents (file_name, extracted_text, chroma_path, timestamp) VALUES (?, ?, ?, ?)",
            (file_name, extracted_text, vector_db_path, timestamp)
        )
        conn.commit()
        logger.info(f"Stored document in SQLite: {file_name}")
    except Exception as e:
        logger.error(f"Failed to store document in SQLite: {e}")
        raise
    finally:
        conn.close()

def load_documents_from_db():
    """Load all documents, their Chroma DB paths, and timestamps from the database."""
    try:
        conn = sqlite3.connect("nalco_chatbot.db")
        cursor = conn.cursor()
        cursor.execute("SELECT file_name, extracted_text, chroma_path, timestamp FROM documents")
        rows = cursor.fetchall()
        logger.info(f"Loaded {len(rows)} documents from SQLite database.")
        return rows
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            logger.error("SQLite table 'documents' does not exist. Please initialize the database.")
            return []  # Return empty list to avoid crashing
        else:
            logger.error(f"Failed to load documents from SQLite: {e}")
            raise
    except Exception as e:
        logger.error(f"Failed to load documents from SQLite: {e}")
        raise
    finally:
        conn.close()