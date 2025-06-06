from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# Import LLM instances from models.py
from models import llm, evaluation_llm
# Import prompt templates and roles from config.py
from config import (
    EVALUATION_PROMPT_SYSTEM,
    MAIN_PROMPT_SYSTEM,
    CHAT_ROLE_USER,
    CHAT_ROLE_AI,
    EVALUATION_EXPLANATION_SYSTEM
)
import json  # For parsing combined evaluation output

# --- Evaluation Chain Setup --- #

# Use the imported template string
EVALUATION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", EVALUATION_PROMPT_SYSTEM),
])

evaluation_chain = (
    EVALUATION_PROMPT_TEMPLATE
    | evaluation_llm # Use the evaluation LLM instance
    | StrOutputParser()
) if evaluation_llm else None # Only define if LLM is available

# --- Main Conversational Chain Setup --- #

# Use the imported template string
MAIN_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", MAIN_PROMPT_SYSTEM),
    (CHAT_ROLE_USER, "{user_message}") # Use role constant
])

def format_chat_history(chat_history: list) -> str:
    """Formats chat history messages into a string for the prompt."""
    if not chat_history:
        return "No history yet."
    # Use role constants
    return "\n".join([f"{msg['role'].replace(CHAT_ROLE_USER, 'Human').replace(CHAT_ROLE_AI, 'AI').capitalize()}: {msg['content']}" for msg in chat_history])

def get_language_level(x):
    # Pass through the language_level from input, default to 3 if missing
    return x.get('language_level', 3)

main_chain = (
    RunnablePassthrough.assign(
        retrieved_context=lambda x: "\n".join([f"- {term['term']}: {term['definition']}" for term in x.get('retrieved_terms', [])]) or "N/A",
        chat_history=lambda x: format_chat_history(x.get('chat_history', [])),
        language_level=get_language_level
    )
    | MAIN_PROMPT_TEMPLATE
    | llm # Use the main LLM instance
    | StrOutputParser()
) if llm else None # Only define chain if LLM is available

# --- Evaluation Function --- #

async def evaluate_turn_success(topic: str, user_message: str, retrieved_context: str) -> str:
    """Calls the LLM evaluation chain to evaluate the user's message."""
    if not evaluation_chain:
        print("Evaluation chain not available. Skipping evaluation.")
        return "no_change"

    try:
        input_data = {
            "topic": topic,
            "user_message": user_message,
            "retrieved_context": retrieved_context if retrieved_context else "None provided."
        }
        result = await evaluation_chain.ainvoke(input_data)
        result = result.strip().lower()

        if result in ["progress", "setback", "no_change"]:
            return result
        else:
            print(f"Warning: Unexpected evaluation result: '{result}'. Defaulting to 'no_change'.")
            return "no_change"

    except Exception as e:
        print(f"Error during LLM evaluation: {e}")
        return "no_change"

# --- Evaluation Explanation Chain Setup --- #

from langchain.prompts import ChatPromptTemplate as _ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser as _StrOutputParser

EVALUATION_EXPLANATION_PROMPT_TEMPLATE = _ChatPromptTemplate.from_messages([
    ("system", EVALUATION_EXPLANATION_SYSTEM),
])

evaluation_explanation_chain = (
    EVALUATION_EXPLANATION_PROMPT_TEMPLATE
    | evaluation_llm
    | _StrOutputParser()
) if evaluation_llm else None

async def explain_evaluation(topic: str, user_message: str, retrieved_context: str) -> str:
    """Returns a 2-3 sentence explanation for the evaluation result."""
    if not evaluation_explanation_chain:
        return ""
    input_data = {
        "topic": topic,
        "user_message": user_message,
        "retrieved_context": retrieved_context if retrieved_context else "None provided."
    }
    try:
        explanation = await evaluation_explanation_chain.ainvoke(input_data)
        return explanation.strip()
    except Exception as e:
        print(f"Error during evaluation explanation: {e}")
        return ""

# --- Combined Evaluation & Explanation Chain Setup --- #
from config import EVALUATION_AND_EXPLANATION_SYSTEM
from langchain.prompts import ChatPromptTemplate as _ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser as _StrOutputParser

EVALUATION_AND_EXPLANATION_PROMPT_TEMPLATE = _ChatPromptTemplate.from_messages([
    ("system", EVALUATION_AND_EXPLANATION_SYSTEM),
])
evaluation_and_explanation_chain = (
    EVALUATION_AND_EXPLANATION_PROMPT_TEMPLATE
    | evaluation_llm
    | _StrOutputParser()
) if evaluation_llm else None

# In-memory cache for combined evaluation and explanation
_eval_and_explain_cache: dict[tuple[str, str, str], dict] = {}

async def evaluate_and_explain(topic: str, user_message: str, retrieved_context: str) -> dict:
    """Performs a single LLM call to get both classification and explanation as JSON."""
    # Check cache first
    cache_key = (topic, user_message, retrieved_context)
    if cache_key in _eval_and_explain_cache:
        return _eval_and_explain_cache[cache_key]
    if not evaluation_and_explanation_chain:
        return {"result": "no_change", "explanation": ""}
    input_data = {
        "topic": topic,
        "user_message": user_message,
        "retrieved_context": retrieved_context if retrieved_context else "None provided."
    }
    try:
        response = await evaluation_and_explanation_chain.ainvoke(input_data)
        parsed = json.loads(response)
        # Ensure keys exist
        result = parsed.get("result", "no_change").strip().lower()
        explanation = parsed.get("explanation", "").strip()
        if result not in ["progress", "setback", "no_change"]:
            result = "no_change"
        result_dict = {"result": result, "explanation": explanation}
        # Cache and return
        _eval_and_explain_cache[cache_key] = result_dict
        return result_dict
    except Exception as e:
        print(f"Error in evaluate_and_explain: {e}")
        return {"result": "no_change", "explanation": ""}
