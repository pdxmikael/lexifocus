from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import os
from config import OPENAI_API_KEY, OPENAI_MODEL_NAME

# Embedding model setup
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Get embedding dimension dynamically
EMBEDDING_DIM = len(embedding_model.embed_query("test"))
print(f"Embedding model loaded. Dimension: {EMBEDDING_DIM}")

# LLM Setup
llm = None
evaluation_llm = None
try:
    # Check if the API key is available
    if OPENAI_API_KEY:
        llm = ChatOpenAI(
            model=OPENAI_MODEL_NAME,
            temperature=0.7
        )
        evaluation_llm = llm # Use the same LLM for both for now
        print(f"Using OpenAI model: {llm.model_name}")
    else:
        print("Warning: OPENAI_API_KEY not found in config/environment. LLM features will be disabled.")
except ImportError:
    print("Warning: langchain_openai not installed. LLM features will be disabled.")
    print("Install with: pip install langchain_openai")
except Exception as e:
    print(f"Error initializing OpenAI LLM: {e}")
