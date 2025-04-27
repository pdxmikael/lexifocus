import chainlit as cl
import os
from dotenv import load_dotenv
import sqlite3 # Ensure sqlite3 is imported
import os # Ensure os is imported
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

    # Create domain_embeddings table - Verify this part
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS domain_embeddings (
        term_id INTEGER PRIMARY KEY AUTOINCREMENT,
        term TEXT NOT NULL UNIQUE,
        definition TEXT NOT NULL,
        topic TEXT NOT NULL, -- Make sure this line is present
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
    """Loads terms and their topics from the YAML file (using existing structure),
       generates embeddings, and stores them in the DB."""
    if not os.path.exists(TERMS_YAML_PATH):
        print(f"Error: Terms file not found at {TERMS_YAML_PATH}")
        return

    try:
        with open(TERMS_YAML_PATH, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading or parsing YAML file: {e}")
        return

    # Validate the structure
    if not data or not isinstance(data, dict) or 'topics' not in data or 'terms' not in data:
        print("Error: YAML file is missing 'topics' or 'terms' sections, or is not structured correctly.")
        return
    if not isinstance(data['topics'], dict) or not isinstance(data['terms'], dict):
         print("Error: 'topics' and 'terms' sections must be dictionaries.")
         return

    all_terms_data = data['terms']
    topics_data = data['topics']

    terms_to_insert = [] # List to hold dicts: {'term_key': key, 'term_sv': sv, 'def_sv': def, 'topic': topic_key}
    texts_to_embed = [] # List to hold strings for embedding: "term_sv: def_sv"
    processed_term_keys = set() # Keep track of term keys for which embeddings are generated

    # Iterate through topics to link terms to topics
    for topic_key, topic_info in topics_data.items():
        if not isinstance(topic_info, dict) or 'terms' not in topic_info or not isinstance(topic_info['terms'], list):
            print(f"Warning: Skipping topic '{topic_key}' due to missing or invalid 'terms' list.")
            continue

        for term_key in topic_info['terms']:
            if term_key not in all_terms_data:
                print(f"Warning: Term key '{term_key}' listed under topic '{topic_key}' not found in the main 'terms' section. Skipping.")
                continue

            term_details = all_terms_data[term_key]
            if not isinstance(term_details, dict) or 'term_sv' not in term_details or 'definition_sv' not in term_details:
                print(f"Warning: Skipping term key '{term_key}' due to missing 'term_sv' or 'definition_sv'.")
                continue

            term_sv = term_details['term_sv']
            definition_sv = term_details['definition_sv']

            # Prepare data for insertion (associating this term instance with this topic)
            # Note: The DB schema currently enforces UNIQUE(term). If a term_sv needs to exist
            # under multiple topics in the DB, the schema needs changing (e.g., remove UNIQUE,
            # or use a separate mapping table). For now, we insert the term with the *first*
            # topic encountered and generate embedding only once per term_key.
            term_info_for_insert = {
                'term_key': term_key, # Store the key for reference
                'term_sv': term_sv,
                'def_sv': definition_sv,
                'topic': topic_key # Use the topic key from the topics section
            }
            terms_to_insert.append(term_info_for_insert)

            # Generate embedding only once per unique term key
            if term_key not in processed_term_keys:
                text_for_embedding = f"{term_sv}: {definition_sv}"
                texts_to_embed.append(text_for_embedding)
                processed_term_keys.add(term_key)
                # Store the index mapping for later retrieval
                term_info_for_insert['embedding_index'] = len(texts_to_embed) - 1


    if not texts_to_embed:
        print("No terms found to process and embed in the YAML file.")
        return

    print(f"Generating embeddings for {len(texts_to_embed)} unique terms...")
    try:
        embeddings = embedding_model.embed_documents(texts_to_embed)
        # Convert embeddings to bytes for SQLite storage
        embedding_blobs = [np.array(emb, dtype=np.float32).tobytes() for emb in embeddings]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    inserted_count = 0
    skipped_count = 0
    # Create a map from term_key to its embedding blob
    term_key_to_embedding = {}
    temp_processed_keys = set() # Need this again for mapping
    embedding_idx_counter = 0
    for term_info in terms_to_insert:
        if 'embedding_index' in term_info and term_info['term_key'] not in temp_processed_keys:
             term_key_to_embedding[term_info['term_key']] = embedding_blobs[embedding_idx_counter]
             temp_processed_keys.add(term_info['term_key'])
             embedding_idx_counter += 1


    # Insert terms, linking the correct embedding blob
    inserted_term_sv = set() # Track inserted term_sv values due to UNIQUE constraint
    for term_info in terms_to_insert:
        term_key = term_info['term_key']
        term_sv = term_info['term_sv']
        definition_sv = term_info['def_sv']
        topic = term_info['topic']
        embedding_blob = term_key_to_embedding.get(term_key) # Get the pre-generated blob

        if not embedding_blob:
             print(f"Error: Could not find embedding for term key '{term_key}'. Skipping insertion.")
             skipped_count += 1
             continue

        # Only insert if the term_sv hasn't been inserted yet (due to UNIQUE constraint)
        if term_sv not in inserted_term_sv:
            try:
                cursor.execute("""
                    INSERT INTO domain_embeddings (term, definition, topic, embedding)
                    VALUES (?, ?, ?, ?)
                """, (term_sv, definition_sv, topic, embedding_blob))
                inserted_term_sv.add(term_sv) # Mark this term_sv as inserted
                inserted_count += 1
            except sqlite3.IntegrityError:
                # This specific term_sv already exists, likely inserted via a different term_key
                # or a previous run. We respect the UNIQUE constraint.
                print(f"Info: Term '{term_sv}' already exists in DB. Skipping duplicate insertion for topic '{topic}'.")
                inserted_term_sv.add(term_sv) # Still mark as 'handled' for this run
                skipped_count += 1
            except sqlite3.Error as e:
                print(f"Error inserting term '{term_sv}' (key: {term_key}): {e}")
                skipped_count += 1
        else:
            # This term_sv was already inserted in *this run* (likely via a different topic association)
            # We skip inserting it again.
            # print(f"Info: Term '{term_sv}' already inserted in this run. Skipping duplicate insertion for topic '{topic}'.")
            skipped_count += 1


    conn.commit()
    conn.close()
    print(f"Term loading complete. Inserted: {inserted_count}, Skipped (already exist, error, or duplicate topic link): {skipped_count}")

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


# LLM Setup (Using the same one for evaluation and main chat for now)
llm = None
try:
    # Check if the API key is available
    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), # Use model from env or default
            temperature=0.7 # Adjust temperature for conversational creativity
        )
        print(f"Using OpenAI model: {llm.model_name}")
    else:
        print("Warning: OPENAI_API_KEY not found. LLM features will be disabled.")
except ImportError:
    print("Warning: langchain_openai not installed. LLM features will be disabled.")
    print("Install with: pip install langchain_openai")
except Exception as e:
    print(f"Error initializing OpenAI LLM: {e}")

# Rename evaluation_llm to llm for clarity if using the same instance
evaluation_llm = llm # Keep evaluation_llm variable if needed elsewhere, pointing to the same object

# --- Evaluation Chain Setup ---

EVALUATION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are an evaluator assessing a student's understanding of a specific Swedish economics topic based on their latest message in a conversation.
The student is learning Swedish economics terms.
The current focus topic is: **'{topic}'**.

Analyze the student's message below in the context of the conversation and the provided relevant terms/definitions.
Determine if the student's message demonstrates **progress**, **setback**, or **no_change** in understanding or correctly using terms related to the focus topic.

- **progress:** The student correctly uses a relevant term, asks a relevant question showing understanding, or discusses the topic accurately in Swedish.
- **setback:** The student misuses a term, shows a clear misunderstanding of the topic, or struggles significantly despite context.
- **no_change:** The message is unrelated, too simple to judge, or shows neither clear progress nor setback (e.g., simple greetings, off-topic remarks).

Retrieved Swedish Terms/Definitions for Context (if any):
{retrieved_context}

Student's Message:
{user_message}

Respond ONLY with "progress", "setback", or "no_change".
"""),
])

evaluation_chain = (
    EVALUATION_PROMPT_TEMPLATE
    | evaluation_llm # Use the dedicated evaluation LLM instance
    | StrOutputParser()
) if evaluation_llm else None # Only define if LLM is available


# --- Main Conversational Chain Setup ---

MAIN_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are LexiFocus, an expert and friendly language tutor specializing in Swedish economics terms for an English-speaking student.
Your goal is to help the student learn and practice these terms through natural conversation.

Instructions:
- Engage the student in a conversation related to economics or finance.
- **Focus Topic:** Try to steer the conversation towards the current focus topic: **'{focus_topic}'**. (If no topic is provided, choose a general economics theme).
- **Use Swedish Primarily:** Conduct most of the conversation in Swedish.
- **Introduce Terms Contextually:** Use the retrieved Swedish terms and definitions below naturally in your Swedish responses when relevant to the conversation. Don't just list them.
- **Switch to English for Clarity:** Explain complex concepts, provide corrections, or clarify nuances of Swedish terms in English when necessary for understanding. Switch back to Swedish afterward.
- **Be Encouraging:** Maintain a positive and supportive tone.
- **Ask Follow-up Questions:** Encourage the student to respond and use the vocabulary.
- **Consider Past Performance:** (Note: The student's previous turn was evaluated as: '{evaluation_feedback}'. Use this subtly, e.g., if 'setback', maybe simplify slightly or offer more support; if 'progress', acknowledge it implicitly or introduce a related term).

Retrieved Swedish Terms/Definitions (Context):
{retrieved_context}

Chat History:
{chat_history}
"""),
    ("human", "{user_message}")
])

# Main Conversational Chain
# This chain will combine context, history, and user message to generate a response
# Note: 'focus_topic' and 'evaluation_feedback' are placeholders for now
#       'chat_history' will be manually formatted and passed in on_message

def format_chat_history(chat_history: list) -> str:
    """Formats chat history messages into a string for the prompt."""
    if not chat_history:
        return "No history yet."
    # Format: "Human: message\nAI: message\n..."
    return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])

main_chain = (
    RunnablePassthrough.assign(
        # Format retrieved context for the prompt
        retrieved_context=lambda x: "\n".join([f"- {term['term']}: {term['definition']}" for term in x.get('retrieved_terms', [])]) or "N/A",
        # Format chat history
        chat_history=lambda x: format_chat_history(x.get('chat_history', []))
    )
    | MAIN_PROMPT_TEMPLATE
    | llm
    | StrOutputParser()
) if llm else None # Only define chain if LLM is available

# Function to evaluate turn success (Uses evaluation_chain now)
async def evaluate_turn_success(topic: str, user_message: str, retrieved_context: str) -> str:
    """Calls the LLM to evaluate the user's message for progress on a topic."""
    if not evaluation_chain: # Check if the chain is defined
        print("Evaluation chain not available. Skipping evaluation.")
        return "no_change" # Default if LLM/chain is not configured

    try:
        input_data = {
            "topic": topic,
            "user_message": user_message,
            "retrieved_context": retrieved_context if retrieved_context else "None provided."
        }
        # Use invoke for synchronous-style call within async function for simplicity here
        # For heavy load, consider async invoke (ainvoke)
        result = await evaluation_chain.ainvoke(input_data) # Use the defined evaluation_chain
        result = result.strip().lower()

        # Validate result
        if result in ["progress", "setback", "no_change"]:
            # print(f"Evaluation result for topic '{topic}': {result}") # Already printed in main loop
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
    user_message_content = message.content
    print(f"\n--- Turn Start ---")
    print(f"User message: {user_message_content}")

    # --- Get Chat History ---
    # Chainlit stores history in the user session
    chat_history = cl.user_session.get("chat_history", [])

    # --- 1. Retrieve Relevant Terms ---
    retrieved_terms = retrieve_relevant_terms(user_message_content, top_n=3, similarity_threshold=0.25) # Adjusted threshold slightly
    retrieved_context_str = "\n".join([f"- {t['term']} ({t['topic']}): {t['definition']} (Similarity: {t['similarity']:.2f})" for t in retrieved_terms]) if retrieved_terms else "No relevant terms found."
    print(f"Retrieved Context:\n{retrieved_context_str}")

    # --- TODO: Step 16/19 - Select Topic ---
    # For now, use a placeholder or the topic from the most relevant retrieved term
    selected_topic_for_turn = retrieved_terms[0]['topic'] if retrieved_terms else "Okänt Ämne" # Placeholder topic selection
    print(f"Selected Topic (Placeholder): {selected_topic_for_turn}")

    # --- 2. Evaluate Turn Success ---
    evaluation_result = "evaluation_disabled" # Default if LLM is off
    if evaluation_llm: # Check if evaluation LLM is available
        evaluation_result = await evaluate_turn_success(
            topic=selected_topic_for_turn,
            user_message=user_message_content,
            retrieved_context=retrieved_context_str # Pass the formatted string
        )
    print(f"Evaluation Result: {evaluation_result}")

    # --- 3. Log Outcome ---
    log_success = evaluation_result == "progress"
    activity_log(topic=selected_topic_for_turn, success=log_success)
    print(f"Activity Logged: Topic='{selected_topic_for_turn}', Success={log_success}")

    # --- TODO: Step 12 - Incorporate Evaluation Feedback ---
    # Placeholder for feedback based on evaluation_result
    evaluation_feedback_for_prompt = evaluation_result # Simple pass-through for now
    print(f"Evaluation Feedback (for next prompt): {evaluation_feedback_for_prompt}")


    # --- 4. Generate Main Response using the Chain ---
    final_response = "Sorry, the main chat functionality is currently disabled as the LLM is not configured." # Default response
    if main_chain: # Check if main chain is available
        try:
            print("Invoking main conversational chain...")
            # Prepare inputs for the main chain
            chain_input = {
                "user_message": user_message_content,
                "retrieved_terms": retrieved_terms, # Pass the list of dicts
                "chat_history": chat_history, # Pass history from session
                "focus_topic": selected_topic_for_turn, # Pass selected topic
                "evaluation_feedback": evaluation_feedback_for_prompt # Pass feedback
            }

            # Stream the response
            msg = cl.Message(content="")
            await msg.send()

            async for chunk in main_chain.astream(chain_input):
                await msg.stream_token(chunk)

            final_response = msg.content # Store final response for history
            await msg.update() # Final update for the streamed message

        except Exception as e:
            print(f"Error invoking main chain: {e}")
            final_response = "An error occurred while generating the response."
            await cl.Message(content=final_response).send()
    else:
        # Send the default message if LLM/chain is disabled
        await cl.Message(content=final_response).send()

    # --- Update Chat History ---
    # Append user message and AI response to history stored in session
    chat_history.append({"role": "human", "content": user_message_content})
    chat_history.append({"role": "ai", "content": final_response})
    cl.user_session.set("chat_history", chat_history)

    print(f"AI Response: {final_response}")
    print(f"--- Turn End ---")
