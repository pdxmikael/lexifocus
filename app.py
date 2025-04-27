import os
import chainlit as cl
from chainlit.action import Action # Import Action
from chainlit.input_widget import Slider, Select
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
    TOPIC_LABEL,  # Label for the topic header
    DEFAULT_VERBOSITY_LEVEL,
    STYLE_OPTIONS
)
from database import init_db, activity_log, retrieve_relevant_terms, get_progress_summary, get_all_topics
from topic_selector import select_topic_thompson, update_bandit_model  # Use refactored module
from data_loader import load_terms_from_yaml
from models import llm, evaluation_llm # Import LLM instances
from chains import main_chain, evaluate_and_explain  # Combined evaluation + explanation helper

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
    """Initializes the chat session and adds actions and chat settings."""
    cl.user_session.set(SESSION_KEY_CHAT_HISTORY, [])
    cl.user_session.set(SESSION_KEY_LAST_EVAL_FEEDBACK, INITIAL_EVAL_FEEDBACK)

    # --- Add ChatSettings for verbosity and style --- #
    settings = cl.ChatSettings([
        Slider(
            id="verbosity_level",
            label="Verbosity",
            min=1,
            max=5,
            step=1,
            initial=DEFAULT_VERBOSITY_LEVEL,
            description="How detailed should the AI's responses be? (1=concise, 5=detailed)"
        ),
        Select(
            id="response_style",
            label="Response Style",
            values=STYLE_OPTIONS,
            initial_index=1,  # Default to 'friendly'
            description="Choose the tone of the AI's responses."
        )
    ])
    await settings.send()

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

@cl.on_settings_update
async def on_settings_update(settings: dict):
    cl.user_session.set("chat_settings", settings)

@cl.on_message
async def main(message: cl.Message):
    settings = cl.user_session.get("chat_settings", {})
    print(f"Received settings: {settings}")
    debug_log(f"--- SETTINGS DICT ---\n{settings}")
    verbosity_level = settings.get("verbosity_level", DEFAULT_VERBOSITY_LEVEL)
    response_style = settings.get("response_style", STYLE_OPTIONS[1])

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
    # --- Adaptive Topic Selection via Thompson Sampling (Step 17) ---
    selected_topic_for_turn = select_topic_thompson()
    # Store selected topic in session
    cl.user_session.set(SESSION_KEY_CURRENT_TOPIC, selected_topic_for_turn)
    print(f"Selected Topic (Thompson Sampling): {selected_topic_for_turn}")

    # --- 2. Evaluate Current Turn Success & Explanation --- #
    evaluation = {"result": DEFAULT_EVALUATION_RESULT, "explanation": ""}
    if evaluation_llm:
        evaluation = await evaluate_and_explain(
            topic=selected_topic_for_turn,
            user_message=user_message_content,
            retrieved_context=retrieved_context_str
        )
    current_evaluation_result = evaluation.get("result", DEFAULT_EVALUATION_RESULT)
    evaluation_explanation = evaluation.get("explanation", "")
    print(f"Current Turn Evaluation Result: {current_evaluation_result}")
    print(f"Evaluation Explanation: {evaluation_explanation}")

    # --- 3. Log Outcome --- #
    log_success = current_evaluation_result == "progress"
    activity_log(topic=selected_topic_for_turn, success=log_success)
    print(f"Activity Logged: Topic='{selected_topic_for_turn}', Success={log_success}")
    # --- 4. Update Bandit Model (Step 18) ---
    update_bandit_model(selected_topic_for_turn, log_success)

    # --- Store Current Evaluation for *Next* Turn --- #
    cl.user_session.set(SESSION_KEY_LAST_EVAL_FEEDBACK, current_evaluation_result)

    # --- 4. Generate Main Response using the Chain --- #
    final_response = LLM_DISABLED_MESSAGE
    # Only include progress action by default
    actions_for_response = [progress_action]

    if main_chain: # Check if the main chain instance exists
        try:
            print("Invoking main conversational chain...")
            chain_input = {
                "user_message": user_message_content,
                "retrieved_terms": retrieved_terms,
                "chat_history": chat_history,
                "focus_topic": selected_topic_for_turn,
                "evaluation_feedback": previous_evaluation_feedback,
                "evaluation_explanation": evaluation_explanation,
                "verbosity_level": verbosity_level,
                "response_style": response_style
            }
            # --- Debug log the system prompt and input ---
            from chains import MAIN_PROMPT_SYSTEM
            debug_log(f"\n--- SYSTEM PROMPT ---\n{MAIN_PROMPT_SYSTEM.format(**{k: chain_input.get(k, '') for k in ['focus_topic','verbosity_level','response_style','evaluation_feedback','evaluation_explanation','retrieved_context','chat_history']})}")
            debug_log(f"--- USER MESSAGE ---\n{user_message_content}")

            # Add evaluation feedback banner if available
            if current_evaluation_result in ("progress", "setback"):
                stats = get_progress_summary().get(selected_topic_for_turn, {})
                succ = stats.get("successful_attempts", 0)
                tot = stats.get("total_attempts", 0)
                rate = stats.get("success_rate", 0)
                emoji = "✅" if current_evaluation_result == "progress" else "❌"
                color = "green" if current_evaluation_result == "progress" else "red"
                # Prepare plain Markdown feedback with emoji and bold
                status_text = 'Progress made' if current_evaluation_result == 'progress' else 'Setback'
                feedback_line = (
                    f"{emoji} **{status_text} in {selected_topic_for_turn}!** "
                    f"Current progress: {succ}/{tot} ({rate}% success)\n\n"
                )
                await cl.Message(content=feedback_line).send()

            # Add prominent topic header to response
            topic_header = f"**{TOPIC_LABEL}:** {selected_topic_for_turn}\n\n"
            msg = cl.Message(content=topic_header, actions=[progress_action])
            await msg.send()

            async for chunk in main_chain.astream(chain_input):
                await msg.stream_token(chunk)
            final_response = msg.content
            debug_log(f"--- LLM RESPONSE ---\n{final_response}")

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

def debug_log(message: str):
    if os.getenv('LEXIFOCUS_DEBUG') == '1':
        with open('debug_log.txt', 'a', encoding='utf-8') as f:
            f.write(message + '\n')
