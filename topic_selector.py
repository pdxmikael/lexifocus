import random
import sqlite3
from config import DB_PATH
from database import get_bandit_params, get_all_topics


def select_topic_thompson() -> str:
    """Selects a topic by Thompson Sampling over alpha/beta parameters."""
    params = get_bandit_params()
    # Sample from Beta distribution for each topic
    samples = {topic: random.betavariate(alpha, beta) for topic, (alpha, beta) in params.items()}
    # Return topic with highest sample
    return max(samples, key=samples.get)


def update_bandit_model(topic: str, success: bool):
    """Updates the bandit parameters for a topic based on outcome."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if success:
        cursor.execute(
            "UPDATE bandit_params SET alpha = alpha + 1 WHERE topic = ?",
            (topic,)
        )
    else:
        cursor.execute(
            "UPDATE bandit_params SET beta = beta + 1 WHERE topic = ?",
            (topic,)
        )
    conn.commit()
    conn.close()