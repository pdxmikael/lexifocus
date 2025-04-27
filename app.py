import chainlit as cl
import os
from dotenv import load_dotenv
import sqlite3
import datetime
import yaml
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables (optional, but good practice)
load_dotenv()

# Database setup
DB_PATH = os.path.join("database", "lexifocus.db")
TERMS_YAML_PATH = os.path.join("data", "terms.yaml") # Added

# Embedding model setup (using a common sentence transformer model)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Get embedding dimension (important for deserializing blobs) - Added
EMBEDDING_DIM = len(embedding_model.embed_query("test")) # Added

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

# Function to load terms and embeddings into the database
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
        # Ensure the dtype matches the dimension and type used for storage
        embedding_blob = np.array(embedding_vector, dtype=np.float32).tobytes() # Added dtype=np.float32

        try:
            # Use INSERT OR IGNORE to avoid errors if the term already exists (based on UNIQUE constraint)
            cursor.execute("""
            INSERT OR IGNORE INTO domain_embeddings (term, definition, embedding)
            VALUES (?, ?, ?)
            """, (term_sv, definition_sv, embedding_blob))
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


# Function to retrieve relevant terms based on message similarity
def retrieve_relevant_terms(user_message_text: str, top_n: int = 3, similarity_threshold: float = 0.3) -> list[dict]: # Added threshold
    """Retrieves the top_n most relevant terms from the DB based on semantic similarity, above a threshold."""
    if not user_message_text:
        return []

    try:
        # 1. Generate embedding for the user message
        message_embedding = embedding_model.embed_query(user_message_text)
        message_embedding_np = np.array(message_embedding, dtype=np.float32).reshape(1, -1)

        # 2. Fetch all term embeddings from the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT term, definition, embedding FROM domain_embeddings")
        all_terms_data = cursor.fetchall()
        conn.close()

        if not all_terms_data:
            print("No terms found in the database for retrieval.")
            return []

        # 3. Deserialize embeddings and prepare for similarity calculation
        term_texts = []
        term_definitions = []
        term_embeddings_np = []
        for term, definition, embedding_blob in all_terms_data:
            # Deserialize BLOB back to numpy array
            term_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            # Ensure the embedding has the correct shape/dimension
            if term_embedding.size == EMBEDDING_DIM:
                term_texts.append(term)
                term_definitions.append(definition)
                term_embeddings_np.append(term_embedding)
            else:
                print(f"Warning: Skipping term '{term}' due to unexpected embedding dimension ({term_embedding.size} vs {EMBEDDING_DIM}).")

        if not term_embeddings_np:
            print("No valid embeddings found after deserialization.")
            return []

        term_embeddings_matrix = np.vstack(term_embeddings_np)

        # 4. Calculate cosine similarity
        similarities = cosine_similarity(message_embedding_np, term_embeddings_matrix)[0]

        # 5. Get indices and scores above the threshold
        relevant_indices_scores = [
            (index, score) for index, score in enumerate(similarities) if score >= similarity_threshold
        ]

        # 6. Sort by score and take top N
        relevant_indices_scores.sort(key=lambda item: item[1], reverse=True)
        top_matches = relevant_indices_scores[:top_n]

        # 7. Retrieve the corresponding terms and definitions
        relevant_terms = []
        for index, score in top_matches:
            relevant_terms.append({
                "term": term_texts[index],
                "definition": term_definitions[index],
                "similarity": float(score) # Use the actual score
            })

        return relevant_terms

    except sqlite3.Error as e:
        print(f"Database error during retrieval: {e}")
        return []
    except Exception as e:
        print(f"Error during term retrieval: {e}")
        return []


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

    # 1. Retrieve relevant terms (Added)
    relevant_terms = retrieve_relevant_terms(message.content, top_n=3)

    # 2. Format retrieved terms for display/context (Added)
    retrieved_context = "No specific terms found relevant to your message."
    if relevant_terms:
        retrieved_context = "Potentially relevant terms based on your message:\n"
        for i, term_info in enumerate(relevant_terms):
            retrieved_context += f"{i+1}. **{term_info['term']}**: {term_info['definition']} (Similarity: {term_info['similarity']:.2f})\n"

    # For now, send the retrieved context back to the user for verification
    # In the next steps, this context will be passed to the LLM
    await cl.Message(
        content=f"--- Debug: Retrieved Context ---\n{retrieved_context}\n---\nOriginal message: {message.content}",
    ).send()

    # Placeholder for actual LLM call using the message and retrieved_context
    # llm_response = call_llm(message.content, retrieved_context)
    # await cl.Message(content=llm_response).send()
