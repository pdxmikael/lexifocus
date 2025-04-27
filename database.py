import sqlite3
import os
import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import DB_PATH
# Import embedding model and dimension from models.py
from models import embedding_model, EMBEDDING_DIM

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
        topic TEXT NOT NULL,
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

def activity_log(topic: str, success: bool):
    """Logs a user interaction outcome to the activity_log table."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        success_int = 1 if success else 0

        cursor.execute("""
        INSERT INTO activity_log (timestamp, topic, success)
        VALUES (?, ?, ?)
        """, (timestamp, topic, success_int))

        conn.commit()
        # print(f"Activity logged: Topic='{topic}', Success={success}") # Keep logging minimal

    except sqlite3.Error as e:
        print(f"Database error during activity logging: {e}")
    except Exception as e:
        print(f"Error during activity logging: {e}")
    finally:
        if conn:
            conn.close()

def retrieve_relevant_terms(user_message_text: str, top_n: int = 3, similarity_threshold: float = 0.3) -> list[dict]:
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
        # Fetch topic along with other data
        cursor.execute("SELECT term, definition, topic, embedding FROM domain_embeddings")
        all_terms_data = cursor.fetchall()
        conn.close()

        if not all_terms_data:
            print("No terms found in the database for retrieval.")
            return []

        # 3. Deserialize embeddings and prepare for similarity calculation
        term_texts = []
        term_definitions = []
        term_topics = [] # Store topics
        term_embeddings_np = []
        for term, definition, topic, embedding_blob in all_terms_data:
            term_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            if term_embedding.size == EMBEDDING_DIM:
                term_texts.append(term)
                term_definitions.append(definition)
                term_topics.append(topic) # Add topic
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

        # 7. Retrieve the corresponding terms, definitions, and topics
        relevant_terms = []
        for index, score in top_matches:
            relevant_terms.append({
                "term": term_texts[index],
                "definition": term_definitions[index],
                "topic": term_topics[index], # Include topic in the result
                "similarity": float(score)
            })

        return relevant_terms

    except sqlite3.Error as e:
        print(f"Database error during retrieval: {e}")
        return []
    except Exception as e:
        print(f"Error during term retrieval: {e}")
        return []

def get_progress_summary() -> dict[str, dict]:
    """Calculates the progress (success rate) for each topic in the activity log."""
    summary = {}
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get total counts and success counts per topic
        cursor.execute("""
            SELECT
                topic,
                COUNT(*) as total_attempts,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_attempts
            FROM activity_log
            GROUP BY topic
        """)
        rows = cursor.fetchall()

        for row in rows:
            topic, total_attempts, successful_attempts = row
            # Ensure successful_attempts is not None (happens if no successes)
            successful_attempts = successful_attempts or 0
            if total_attempts > 0:
                success_rate = (successful_attempts / total_attempts) * 100
            else:
                success_rate = 0

            summary[topic] = {
                "total_attempts": total_attempts,
                "successful_attempts": successful_attempts,
                "success_rate": round(success_rate, 1) # Round to one decimal place
            }

    except sqlite3.Error as e:
        print(f"Database error fetching progress summary: {e}")
        return {} # Return empty dict on error
    except Exception as e:
        print(f"Error fetching progress summary: {e}")
        return {}
    finally:
        if conn:
            conn.close()

    return summary
