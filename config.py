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
