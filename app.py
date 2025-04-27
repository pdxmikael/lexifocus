import chainlit as cl
import os
from dotenv import load_dotenv
import sqlite3
import datetime
import yaml
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI # Re-added

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


# Function to log user activity/progress (Added)
def activity_log(topic: str, success: bool):
    """Logs a user interaction outcome to the activity_log table."""
    conn = None # Initialize conn to None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        # Convert boolean success to integer (1 for True, 0 for False) for SQLite
        success_int = 1 if success else 0

        cursor.execute("""
        INSERT INTO activity_log (timestamp, topic, success)
        VALUES (?, ?, ?)
        """, (timestamp, topic, success_int))

        conn.commit()
        print(f"Activity logged: Topic='{topic}', Success={success}") # Optional: confirmation log

    except sqlite3.Error as e:
        print(f"Database error during activity logging: {e}")
    except Exception as e:
        print(f"Error during activity logging: {e}")
    finally:
        if conn:
            conn.close()


# LLM Setup for Evaluation (Reverted to OpenAI)
# Ensure OPENAI_API_KEY is set in your .env file or environment variables
try:
    evaluation_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # Using OpenAI model
except ImportError:
    print("Error: langchain-openai package not found. Please install it: pip install langchain-openai")
    evaluation_llm = None
except Exception as e:
    print(f"Error initializing OpenAI LLM (check API key?): {e}")
    evaluation_llm = None

# Prompt Template for Evaluation (No changes needed)
EVALUATION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "You are an expert language tutor evaluating a student's understanding of a specific topic based on their latest message in a conversation. The student is learning Swedish economics terms. Their native language is English."),
    ("human", """Analyze the student's latest message regarding the topic: **{topic}**.

Consider the following potentially relevant terms and definitions:
{retrieved_context}

Student's latest message:
'{user_message}'

Based *only* on this single message and the provided context, did the student demonstrate clear progress in understanding or correctly using the term(s) related to '{topic}' in Swedish?

Respond with ONLY ONE of the following words:
- 'progress': If the student showed clear understanding or correct usage related to the topic.
- 'setback': If the student showed clear misunderstanding or incorrect usage related to the topic.
- 'no_change': If the message was too short, irrelevant to the topic, or didn't provide enough information to judge progress or setback.""")
])

# Evaluation Chain (No changes needed, uses the updated evaluation_llm)
evaluation_chain = (
    RunnablePassthrough()
    | EVALUATION_PROMPT_TEMPLATE
    | evaluation_llm
    | StrOutputParser()
) if evaluation_llm else None

# Function to evaluate turn success (No changes needed)
async def evaluate_turn_success(topic: str, user_message: str, retrieved_context: str) -> str:
    """Calls the LLM to evaluate the user's message for progress on a topic."""
    if not evaluation_chain:
        print("Evaluation LLM not available. Skipping evaluation.")
        return "no_change" # Default if LLM is not configured

    try:
        input_data = {
            "topic": topic,
            "user_message": user_message,
            "retrieved_context": retrieved_context if retrieved_context else "None provided."
        }
        # Use invoke for synchronous-style call within async function for simplicity here
        # For heavy load, consider async invoke (ainvoke)
        result = await evaluation_chain.ainvoke(input_data)
        result = result.strip().lower()

        # Validate result
        if result in ["progress", "setback", "no_change"]:
            print(f"Evaluation result for topic '{topic}': {result}")
            return result
        else:
            print(f"Warning: Unexpected evaluation result: '{result}'. Defaulting to 'no_change'.")
            return "no_change"

    except Exception as e:
        print(f"Error during LLM evaluation: {e}")
        return "no_change" # Default on error

# Initialize the database on application startup
init_db()
# Load terms and generate embeddings on startup
load_terms_from_yaml()

@cl.on_chat_start
async def start_chat():
    """Initializes the chat session."""
    await cl.Message(content="Welcome to LexiFocus! Let's start learning.").send()

@cl.on_message
async def main(message: cl.Message):
    """Handles incoming user messages."""
    user_message_content = message.content

    # --- Placeholder for Topic Selection (Step 13) ---
    selected_topic_for_turn = "Inflation" # Using 'Inflation' as a placeholder topic
    # --- End Placeholder ---

    # 1. Retrieve relevant terms
    relevant_terms = retrieve_relevant_terms(user_message_content, top_n=3)
    retrieved_context_str = "No specific terms found relevant to your message."
    if relevant_terms:
        retrieved_context_str = "Potentially relevant terms based on your message:\n"
        for i, term_info in enumerate(relevant_terms):
            retrieved_context_str += f"{i+1}. **{term_info['term']}**: {term_info['definition']} (Similarity: {term_info['similarity']:.2f})\n"

    # --- Debug: Show Retrieved Context ---
    await cl.Message(
        content=f"--- Debug: Retrieved Context ---\n{retrieved_context_str}\n---",
        parent_id=message.id # Indent under the user message
    ).send()
    # --- End Debug ---

    # 2. Evaluate Turn Success
    evaluation_result = await evaluate_turn_success(
        topic=selected_topic_for_turn,
        user_message=user_message_content,
        retrieved_context=retrieved_context_str
    )

    # --- Debug: Show Evaluation Result ---
    await cl.Message(
        content=f"--- Debug: Evaluation Result for topic '{selected_topic_for_turn}' ---\n{evaluation_result}\n---",
        parent_id=message.id # Indent under the user message
    ).send()
    # --- End Debug ---

    # 3. Log Outcome (Added)
    # Convert evaluation result string to boolean for logging
    log_success = evaluation_result == "progress"
    activity_log(topic=selected_topic_for_turn, success=log_success)

    # Placeholder for actual LLM call using the message and retrieved_context
    # llm_response = call_llm(message.content, retrieved_context)
    # await cl.Message(content=llm_response).send()

    # Placeholder response for now
    await cl.Message(
        content=f"Received: '{user_message_content}'. Evaluation: {evaluation_result}. Logged: {log_success}. (LLM response not implemented yet)"
    ).send()
