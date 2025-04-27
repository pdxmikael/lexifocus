from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# Import LLM instances from models.py
from models import llm, evaluation_llm

# --- Evaluation Chain Setup --- #

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
    | evaluation_llm # Use the evaluation LLM instance
    | StrOutputParser()
) if evaluation_llm else None # Only define if LLM is available

# --- Main Conversational Chain Setup --- #

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

def format_chat_history(chat_history: list) -> str:
    """Formats chat history messages into a string for the prompt."""
    if not chat_history:
        return "No history yet."
    return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])

main_chain = (
    RunnablePassthrough.assign(
        retrieved_context=lambda x: "\n".join([f"- {term['term']}: {term['definition']}" for term in x.get('retrieved_terms', [])]) or "N/A",
        chat_history=lambda x: format_chat_history(x.get('chat_history', []))
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
