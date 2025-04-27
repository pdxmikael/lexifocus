import chainlit as cl
import os
from dotenv import load_dotenv
import sqlite3
import datetime
import yaml # Added
import numpy as np # Added
from langchain_huggingface import HuggingFaceEmbeddings # Added

# Load environment variables (optional, but good practice)
load_dotenv()

# Database setup
DB_PATH = os.path.join("database", "lexifocus.db")
TERMS_YAML_PATH = os.path.join("data", "terms.yaml") # Added

# Embedding model setup (using a common sentence transformer model)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # Added

def init_db():
    """Initializes the SQLite database and creates tables if they don't exist."""
    # Ensure the database directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create domain_embeddings table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS domain_embeddings (
        term_id INTEGER PRIMARY KEY AUTOINCREMENT,
        term TEXT NOT NULL UNIQUE,
        definition TEXT NOT NULL,
        embedding BLOB NOT NULL
    )
    """)

    # Create activity_log table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS activity_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        topic TEXT NOT NULL,
        success BOOLEAN NOT NULL
    )
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

# Function to load terms and embeddings into the database (Added)
def load_terms_from_yaml():
    """Loads terms from the YAML file, generates embeddings, and stores them in the DB."""
    if not os.path.exists(TERMS_YAML_PATH):
        print(f"Warning: Terms file not found at {TERMS_YAML_PATH}")
        return

    try:
        with open(TERMS_YAML_PATH, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading or parsing YAML file {TERMS_YAML_PATH}: {e}")
        return

    if 'terms' not in data or not isinstance(data['terms'], dict):
        print(f"Warning: 'terms' key not found or not a dictionary in {TERMS_YAML_PATH}")
        return

    terms_to_insert = []
    texts_to_embed = []
    term_keys = []

    for key, term_data in data['terms'].items():
        if 'term_sv' in term_data and 'definition_sv' in term_data:
            term_keys.append(key)
            # Embed the Swedish definition
            texts_to_embed.append(term_data['definition_sv'])
            terms_to_insert.append((
                key, # Using the key from YAML as the 'term' column
                term_data['term_sv'],
                term_data['definition_sv']
            ))
        else:
            print(f"Warning: Skipping term '{key}' due to missing 'term_sv' or 'definition_sv'.")

    if not texts_to_embed:
        print("No valid terms found to embed and load.")
        return

    print(f"Generating embeddings for {len(texts_to_embed)} terms...")
    try:
        embeddings = embedding_model.embed_documents(texts_to_embed)
        print("Embeddings generated.")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    inserted_count = 0
    skipped_count = 0
    for i, term_tuple in enumerate(terms_to_insert):
        term_key, term_sv, definition_sv = term_tuple
        embedding_vector = embeddings[i]
        # Convert numpy array/list of floats to bytes for BLOB storage
        embedding_blob = np.array(embedding_vector, dtype=np.float32).tobytes()

        try:
            # Use INSERT OR IGNORE to avoid errors if the term already exists (based on UNIQUE constraint)
            cursor.execute("""
            INSERT OR IGNORE INTO domain_embeddings (term, definition, embedding)
            VALUES (?, ?, ?)
            """, (term_sv, definition_sv, embedding_blob)) # Changed: Using term_sv as the unique term identifier now
            if cursor.rowcount > 0:
                inserted_count += 1
            else:
                skipped_count += 1
        except sqlite3.Error as e:
            print(f"Database error inserting term '{term_sv}': {e}")
            skipped_count += 1 # Count as skipped if error occurs

    conn.commit()
    conn.close()
    print(f"Term loading complete. Inserted: {inserted_count}, Skipped (already exist or error): {skipped_count}")


# Initialize the database on application startup
init_db()
# Load terms and generate embeddings on startup (Added)
load_terms_from_yaml()

@cl.on_chat_start
async def start_chat():
    """Initializes the chat session."""
    await cl.Message(content="Welcome to LexiFocus! Let's start learning.").send()

@cl.on_message
async def main(message: cl.Message):
    """Handles incoming user messages."""
    # For now, just echo the message back
    await cl.Message(
        content=f"You said: {message.content}",
    ).send()
