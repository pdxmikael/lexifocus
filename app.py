import chainlit as cl
from chainlit.action import Action # Import Action

# Import necessary functions and objects from the new modules
from config import (
    OPENAI_API_KEY,  # Check if LLM is configured
    SESSION_KEY_CHAT_HISTORY,
    SESSION_KEY_LAST_EVAL_FEEDBACK,
    SESSION_KEY_CURRENT_TOPIC,  # Current topic session key
    INITIAL_EVAL_FEEDBACK,
    DEFAULT_TOPIC,
    DEFAULT_EVALUATION_RESULT,
    WELCOME_MESSAGE,
    NO_RELEVANT_TERMS_MESSAGE,
    LLM_DISABLED_MESSAGE,
    ERROR_GENERATING_RESPONSE_MESSAGE,
    CHAT_ROLE_USER,
    CHAT_ROLE_AI,
    PROGRESS_ICON_NAME,
    PROGRESS_TOOLTIP,
    TOPIC_LABEL  # Label for the topic header
)
from database import init_db, activity_log, retrieve_relevant_terms, get_progress_summary, get_all_topics  # Added get_all_topics
from data_loader import load_terms_from_yaml
from models import llm, evaluation_llm # Import LLM instances
from chains import main_chain, evaluate_turn_success # Import chains and evaluation function

# --- Application Setup --- #

# Initialize the database on application startup
init_db()
# Load terms and generate embeddings on startup
load_terms_from_yaml()

# --- Chainlit Event Handlers --- #

# Define progress action globally
progress_action = Action(
    name="show_progress",
    value="show",
    icon=PROGRESS_ICON_NAME,
    tooltip=PROGRESS_TOOLTIP,
    payload={}
)

@cl.on_chat_start
async def start_chat():
    """Initializes the chat session and adds actions."""
    cl.user_session.set(SESSION_KEY_CHAT_HISTORY, [])
    cl.user_session.set(SESSION_KEY_LAST_EVAL_FEEDBACK, INITIAL_EVAL_FEEDBACK)
    # Add only the progress action to the welcome message
    await cl.Message(content=WELCOME_MESSAGE, actions=[progress_action]).send()

# --- Action Callback for Progress Button --- #

@cl.action_callback("show_progress")
async def on_show_progress(action: Action):
    """Handles the click event for the 'Show Progress' button."""
    print(f"Action triggered: {action.name}")
    # TODO: Implement Step 14 - Fetch and display actual progress
    progress_summary = get_progress_summary()
    if not progress_summary:
        await cl.Message(content="No progress data available yet.").send()
        return

    # Format the summary for display
    content = "## Your Progress:\n\n"
    for topic, data in progress_summary.items():
        content += f"**{topic}:** {data['successful_attempts']}/{data['total_attempts']} attempts ({data['success_rate']} % success)\n"

    await cl.Message(content=content).send()
    # Remove the button after displaying progress (optional)
    # await action.remove()

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

    # --- Adaptive Topic Selection (Step 15) --- #
    topics = get_all_topics()
    progress_summary = get_progress_summary()
    # Prioritize topics below 80% mastery
    low_mastery = [t for t in topics if progress_summary.get(t, {}).get('success_rate', 0) < 80]
    if low_mastery:
        # Pick the topic with lowest success rate
        selected_topic_for_turn = min(low_mastery, key=lambda t: progress_summary.get(t, {}).get('success_rate', 0))
    else:
        # Round-robin among all topics
        prev_topic = cl.user_session.get(SESSION_KEY_CURRENT_TOPIC)
        if prev_topic in topics:
            idx = topics.index(prev_topic)
            selected_topic_for_turn = topics[(idx + 1) % len(topics)]
        else:
            selected_topic_for_turn = topics[0] if topics else DEFAULT_TOPIC
    # Store selected topic in session
    cl.user_session.set(SESSION_KEY_CURRENT_TOPIC, selected_topic_for_turn)
    print(f"Selected Topic: {selected_topic_for_turn}")

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
    # Only include progress action by default
    actions_for_response = [progress_action]

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

            # Add prominent topic header to response
            topic_header = f"**{TOPIC_LABEL}:** {selected_topic_for_turn}\n\n"
            msg = cl.Message(content=topic_header, actions=[progress_action])
            await msg.send()

            async for chunk in main_chain.astream(chain_input):
                await msg.stream_token(chunk)

            final_response = msg.content

            # Update the message content and actions (only progress action)
            msg.actions = actions_for_response
            await msg.update()

        except Exception as e:
            print(f"Error invoking main chain: {e}")
            final_response = ERROR_GENERATING_RESPONSE_MESSAGE
            # Send error message WITH only progress action
            await cl.Message(content=final_response, actions=actions_for_response).send()
    else:
        # Send disabled message WITH only progress action
        await cl.Message(content=final_response, actions=actions_for_response).send()

    # --- Update Chat History --- #
    chat_history.append({"role": CHAT_ROLE_USER, "content": user_message_content})
    chat_history.append({"role": CHAT_ROLE_AI, "content": final_response})
    cl.user_session.set(SESSION_KEY_CHAT_HISTORY, chat_history)

    print(f"AI Response: {final_response}")
    print(f"--- Turn End ---")
