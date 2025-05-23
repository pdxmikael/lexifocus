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
SESSION_KEY_CURRENT_TOPIC = "current_topic"  # Key for storing the current topic
SESSION_KEY_VERBOSITY = "verbosity_level"  # Key for storing verbosity setting
SESSION_KEY_RESPONSE_STYLE = "response_style"  # Key for storing style setting

# Default/Initial Values
INITIAL_EVAL_FEEDBACK = "N/A"
DEFAULT_TOPIC = "Okänt Ämne" # "Unknown Topic" in Swedish
DEFAULT_EVALUATION_RESULT = "evaluation_disabled"

# User-Facing Messages
WELCOME_MESSAGE = "Välkommen till LexiFocus! Vad intresserar dig idag?"
NO_RELEVANT_TERMS_MESSAGE = "No relevant terms found."
LLM_DISABLED_MESSAGE = "Sorry, the main chat functionality is currently disabled as the LLM is not configured."
ERROR_GENERATING_RESPONSE_MESSAGE = "An error occurred while generating the response."
# PROGRESS_BUTTON_LABEL = "Show Progress" # No longer needed if using icon only
PROGRESS_PLACEHOLDER_MESSAGE = "Progress view is not fully implemented yet."
PROGRESS_ICON_NAME = "bar-chart-horizontal" # Lucide icon name
PROGRESS_TOOLTIP = "Show Progress"  # Tooltip for the icon button

# UI Labels
TOPIC_LABEL = "Ämne"  # Swedish label for the current focus topic header

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

- **progress:** The student correctly uses a relevant term in context or discusses the topic accurately in Swedish.
- **setback:** The student misuses a term, shows a clear misunderstanding of the topic, or struggles significantly despite context.
- **no_change:** The message is unrelated, too simple to judge, shows neither clear progress nor setback, or asks for a term definition without applying it (e.g., simple greetings, off-topic remarks, or definition queries).

Retrieved Swedish Terms/Definitions for Context (if any):
{retrieved_context}

Student's Message:
{user_message}

Respond ONLY with "progress", "setback", or "no_change".
"""

# Explanation Prompt (System Message)
EVALUATION_EXPLANATION_SYSTEM = EVALUATION_PROMPT_SYSTEM + "\n\nPlease provide a 2-3 sentence explanation for why you chose this classification."  

# Combined Evaluation & Explanation Prompt (System Message)
EVALUATION_AND_EXPLANATION_SYSTEM = EVALUATION_PROMPT_SYSTEM + "\n\nProvide your response as a JSON object with two fields:\n- result: one of \"progress\", \"setback\", or \"no_change\"\n- explanation: a 2-3 sentence rationale for your classification\nReturn only the JSON object."

# Main Conversation Prompt (System Message)
MAIN_PROMPT_SYSTEM = """
***IMPORTANT:***
You MUST strictly follow these three response controls for every reply:
- **Verbosity:** Your response MUST match the verbosity level: **{verbosity_level}** (1=very concise, 5=extremely detailed). Do NOT ignore this. If 1, reply in 1-2 short sentences. If 5, reply with several detailed paragraphs, examples, and explanations.
- **Style:** Your response MUST match the style: **{response_style}**. 
    - factual: Be neutral, objective, and direct.
    - friendly: Be warm, encouraging, and conversational.
    - playful: Be lighthearted, use humor or analogies, and make learning fun without being patronizing. Do not say "Haha" or similar unless the user makes a joke.
- **Language Level:** Your response MUST match the language complexity level: **{language_level}** (1=Grade 4, 2=Grade 7, 3=High School, 4=Professional, 5=Academic). Adjust ONLY the general language and sentence structure, NOT the terms being taught. If 1, use simple words and short sentences. If 5, use advanced vocabulary and complex sentences. Do NOT change the target terms or definitions.

If you do not follow these controls, your response will be considered incorrect.

You are LexiFocus, an expert language tutor specializing in Swedish economics terms for an English-speaking student.
Your goal is to help the student learn and practice these terms through natural conversation.

Instructions:
- Engage the student in a conversation related to economics or finance.
- **Focus Topic:** Steer the conversation towards the current focus topic: **'{focus_topic}'**. (If no topic is provided, choose a general economics theme).
- **Use Swedish Primarily:** Conduct most of the conversation in Swedish.
- **Introduce Terms Contextually:** Use the retrieved Swedish terms and their provided definitions naturally in your Swedish responses when relevant to the conversation. Don't just list them.
- **Correct Terminology:** If the user misspells or misformats a term that matches a known vocabulary word, correct it and always use the canonical term in your responses. Correct rather than mirror the user's errors.
- **Switch to English for Clarity:** Explain complex concepts, provide corrections, or clarify nuances of Swedish terms in English when necessary for understanding. Switch back to Swedish afterward.
- **Be Encouraging:** Maintain a positive and supportive tone.
- **Ask Follow-up Questions:** Encourage the student to respond and use the vocabulary. Keep questions specific until the student shows understanding, then broaden the scope.
- **Consider Past Performance:** (Note: The student's previous turn was evaluated as: '{evaluation_feedback}'. Use this subtly, e.g., if 'setback', simplify slightly or offer more support; if 'progress', acknowledge it implicitly or introduce a related term)
**Evaluation Explanation:** Use the following explanation of the most recent evaluation to guide your response: {evaluation_explanation}

Retrieved Swedish Terms/Definitions (Context):
{retrieved_context}

Chat History:
{chat_history}

***REMINDER:*** If you do not match the requested verbosity, style, and language level, your response will be considered incorrect.xx
"""

# Response Mode Defaults and Options
DEFAULT_VERBOSITY_LEVEL = 3  # Scale 1 (concise) to 5 (detailed)
STYLE_OPTIONS = ["factual", "friendly", "playful"]  # Corresponding to slider positions 1–3

# Language Level Options
LANGUAGE_LEVEL_LABELS = [
    "Grade 4",
    "Grade 7",
    "High School",
    "Professional",
    "Academic"
]
