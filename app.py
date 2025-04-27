import chainlit as cl

# Import necessary functions and objects from the new modules
from config import (
    OPENAI_API_KEY, # Check if LLM is configured
    SESSION_KEY_CHAT_HISTORY,
    SESSION_KEY_LAST_EVAL_FEEDBACK,
    INITIAL_EVAL_FEEDBACK,
    DEFAULT_TOPIC,
    DEFAULT_EVALUATION_RESULT,
    WELCOME_MESSAGE,
    NO_RELEVANT_TERMS_MESSAGE,
    LLM_DISABLED_MESSAGE,
    ERROR_GENERATING_RESPONSE_MESSAGE,
    CHAT_ROLE_USER,
    CHAT_ROLE_AI
)
from database import init_db, activity_log, retrieve_relevant_terms
from data_loader import load_terms_from_yaml
from models import llm, evaluation_llm # Import LLM instances
from chains import main_chain, evaluate_turn_success # Import chains and evaluation function

# --- Application Setup --- #

# Initialize the database on application startup
init_db()
# Load terms and generate embeddings on startup
load_terms_from_yaml()

# --- Chainlit Event Handlers --- #

@cl.on_chat_start
async def start_chat():
    """Initializes the chat session."""
    # Initialize chat history and evaluation feedback in the session
    cl.user_session.set(SESSION_KEY_CHAT_HISTORY, [])
    cl.user_session.set(SESSION_KEY_LAST_EVAL_FEEDBACK, INITIAL_EVAL_FEEDBACK) # Initialize feedback
    await cl.Message(content=WELCOME_MESSAGE).send()

@cl.on_message
async def main(message: cl.Message):
    user_message_content = message.content
    print(f"\n--- Turn Start ---")
    print(f"User message: {user_message_content}")

    # --- Get Chat History & Previous Evaluation Feedback --- #
    chat_history = cl.user_session.get(SESSION_KEY_CHAT_HISTORY, [])
    previous_evaluation_feedback = cl.user_session.get(SESSION_KEY_LAST_EVAL_FEEDBACK, INITIAL_EVAL_FEEDBACK)
    print(f"Previous Turn Evaluation Feedback (for current prompt): {previous_evaluation_feedback}")

    # --- 1. Retrieve Relevant Terms --- #
    retrieved_terms = retrieve_relevant_terms(user_message_content, top_n=3, similarity_threshold=0.25)
    # Format context string for logging and evaluation input
    retrieved_context_str = "\n".join([f"- {t['term']} ({t['topic']}): {t['definition']} (Similarity: {t['similarity']:.2f})" for t in retrieved_terms]) if retrieved_terms else NO_RELEVANT_TERMS_MESSAGE
    print(f"Retrieved Context:\n{retrieved_context_str}")

    # --- TODO: Step 16/19 - Select Topic --- #
    # Placeholder topic selection: Use topic from the most relevant term or a default
    selected_topic_for_turn = retrieved_terms[0]['topic'] if retrieved_terms else DEFAULT_TOPIC
    print(f"Selected Topic (Placeholder): {selected_topic_for_turn}")

    # --- 2. Evaluate Current Turn Success --- #
    current_evaluation_result = DEFAULT_EVALUATION_RESULT
    if evaluation_llm: # Check if the evaluation LLM instance exists
        current_evaluation_result = await evaluate_turn_success(
            topic=selected_topic_for_turn,
            user_message=user_message_content,
            retrieved_context=retrieved_context_str
        )
    print(f"Current Turn Evaluation Result: {current_evaluation_result}")

    # --- 3. Log Outcome --- #
    log_success = current_evaluation_result == "progress"
    activity_log(topic=selected_topic_for_turn, success=log_success)
    print(f"Activity Logged: Topic='{selected_topic_for_turn}', Success={log_success}")

    # --- Store Current Evaluation for *Next* Turn --- #
    cl.user_session.set(SESSION_KEY_LAST_EVAL_FEEDBACK, current_evaluation_result)
    print(f"Stored Evaluation Feedback for Next Turn: {current_evaluation_result}")

    # --- 4. Generate Main Response using the Chain --- #
    final_response = LLM_DISABLED_MESSAGE
    if main_chain: # Check if the main chain instance exists
        try:
            print("Invoking main conversational chain...")
            chain_input = {
                "user_message": user_message_content,
                "retrieved_terms": retrieved_terms, # Pass the list of dicts
                "chat_history": chat_history,
                "focus_topic": selected_topic_for_turn,
                "evaluation_feedback": previous_evaluation_feedback # Pass the *previous* turn's feedback
            }

            msg = cl.Message(content="")
            await msg.send()

            async for chunk in main_chain.astream(chain_input):
                await msg.stream_token(chunk)

            final_response = msg.content
            await msg.update()

        except Exception as e:
            print(f"Error invoking main chain: {e}")
            final_response = ERROR_GENERATING_RESPONSE_MESSAGE
            await cl.Message(content=final_response).send()
    else:
        await cl.Message(content=final_response).send()

    # --- Update Chat History --- #
    chat_history.append({"role": CHAT_ROLE_USER, "content": user_message_content})
    chat_history.append({"role": CHAT_ROLE_AI, "content": final_response})
    cl.user_session.set(SESSION_KEY_CHAT_HISTORY, chat_history)

    print(f"AI Response: {final_response}")
    print(f"--- Turn End ---")
