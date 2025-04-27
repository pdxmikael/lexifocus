import os
from dotenv import load_dotenv

# Load environment variables (optional, but good practice)
load_dotenv()

# Database setup
DB_PATH = os.path.join("database", "lexifocus.db")
TERMS_YAML_PATH = os.path.join("data", "terms.yaml")

# Embedding model configuration (dimension needed for deserialization)
# This is determined dynamically in models.py now, but could be hardcoded if known
# EMBEDDING_DIM = 384 # Example for all-MiniLM-L6-v2

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

# --- Application Strings ---

# Session Keys
SESSION_KEY_CHAT_HISTORY = "chat_history"
SESSION_KEY_LAST_EVAL_FEEDBACK = "last_evaluation_feedback"

# Default/Initial Values
INITIAL_EVAL_FEEDBACK = "N/A"
DEFAULT_TOPIC = "Okänt Ämne" # "Unknown Topic" in Swedish
DEFAULT_EVALUATION_RESULT = "evaluation_disabled"

# User-Facing Messages
WELCOME_MESSAGE = "Welcome to LexiFocus! Let's start learning."
NO_RELEVANT_TERMS_MESSAGE = "No relevant terms found."
LLM_DISABLED_MESSAGE = "Sorry, the main chat functionality is currently disabled as the LLM is not configured."
ERROR_GENERATING_RESPONSE_MESSAGE = "An error occurred while generating the response."
PROGRESS_BUTTON_LABEL = "Show Progress"
PROGRESS_PLACEHOLDER_MESSAGE = "Progress view is not fully implemented yet."

# Chat Roles
CHAT_ROLE_USER = "human"
CHAT_ROLE_AI = "ai"

# --- LLM Prompt Templates ---

# Evaluation Prompt (System Message)
EVALUATION_PROMPT_SYSTEM = """You are an evaluator assessing a student's understanding of a specific Swedish economics topic based on their latest message in a conversation.
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
"""

# Main Conversation Prompt (System Message)
MAIN_PROMPT_SYSTEM = """You are LexiFocus, an expert and friendly language tutor specializing in Swedish economics terms for an English-speaking student.
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
"""
