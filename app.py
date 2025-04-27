import chainlit as cl
import os
from dotenv import load_dotenv

# Load environment variables (optional, but good practice)
load_dotenv()

@cl.on_chat_start
async def start_chat():
    """Initializes the chat session."""
    await cl.Message(content="Welcome to LexiFocus! Let's start learning.").send()

@cl.on_message
async def main(message: cl.Message):
    """Handles incoming user messages."""
    # For now, just echo the message back
    await cl.Message(
        content=f"You said: {message.content}",
    ).send()
