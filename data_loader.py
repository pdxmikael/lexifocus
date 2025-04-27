import os
import yaml
import numpy as np
import sqlite3
from config import TERMS_YAML_PATH, DB_PATH
from models import embedding_model # Import the embedding model

def load_terms_from_yaml():
    """Loads terms and their topics from the YAML file,
       generates embeddings, and stores them in the DB."""
    if not os.path.exists(TERMS_YAML_PATH):
        print(f"Error: Terms file not found at {TERMS_YAML_PATH}")
        return

    try:
        with open(TERMS_YAML_PATH, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading or parsing YAML file: {e}")
        return

    # Validate the structure
    if not data or not isinstance(data, dict) or 'topics' not in data or 'terms' not in data:
        print("Error: YAML file is missing 'topics' or 'terms' sections, or is not structured correctly.")
        return
    if not isinstance(data['topics'], dict) or not isinstance(data['terms'], dict):
         print("Error: 'topics' and 'terms' sections must be dictionaries.")
         return

    all_terms_data = data['terms']
    topics_data = data['topics']

    terms_to_insert = []
    texts_to_embed = []
    processed_term_keys = set()
    term_key_to_embedding_index = {} # Map term_key to index in texts_to_embed

    # First pass: Identify unique terms to embed and map term_key to embedding index
    embedding_idx_counter = 0
    for topic_key, topic_info in topics_data.items():
        if not isinstance(topic_info, dict) or 'terms' not in topic_info or not isinstance(topic_info['terms'], list):
            print(f"Warning: Skipping topic '{topic_key}' due to missing or invalid 'terms' list.")
            continue
        for term_key in topic_info['terms']:
            if term_key not in all_terms_data:
                print(f"Warning: Term key '{term_key}' listed under topic '{topic_key}' not found. Skipping.")
                continue
            if term_key not in processed_term_keys:
                term_details = all_terms_data[term_key]
                if isinstance(term_details, dict) and 'term_sv' in term_details and 'definition_sv' in term_details:
                    term_sv = term_details['term_sv']
                    definition_sv = term_details['definition_sv']
                    text_for_embedding = f"{term_sv}: {definition_sv}"
                    texts_to_embed.append(text_for_embedding)
                    term_key_to_embedding_index[term_key] = embedding_idx_counter
                    processed_term_keys.add(term_key)
                    embedding_idx_counter += 1
                else:
                     print(f"Warning: Skipping term key '{term_key}' due to missing 'term_sv' or 'definition_sv'.")

    if not texts_to_embed:
        print("No valid terms found to process and embed in the YAML file.")
        return

    print(f"Generating embeddings for {len(texts_to_embed)} unique terms...")
    try:
        embeddings = embedding_model.embed_documents(texts_to_embed)
        embedding_blobs = [np.array(emb, dtype=np.float32).tobytes() for emb in embeddings]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return

    # Second pass: Prepare data for DB insertion
    for topic_key, topic_info in topics_data.items():
         if not isinstance(topic_info, dict) or 'terms' not in topic_info or not isinstance(topic_info['terms'], list):
             continue # Already warned
         for term_key in topic_info['terms']:
             if term_key in all_terms_data and term_key in term_key_to_embedding_index:
                 term_details = all_terms_data[term_key]
                 term_sv = term_details.get('term_sv')
                 definition_sv = term_details.get('definition_sv')
                 embedding_index = term_key_to_embedding_index[term_key]
                 embedding_blob = embedding_blobs[embedding_index]

                 if term_sv and definition_sv and embedding_blob:
                     terms_to_insert.append({
                         'term_sv': term_sv,
                         'def_sv': definition_sv,
                         'topic': topic_key,
                         'embedding': embedding_blob
                     })

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    inserted_count = 0
    skipped_count = 0
    inserted_term_sv = set() # Track inserted term_sv values due to UNIQUE constraint

    for term_info in terms_to_insert:
        term_sv = term_info['term_sv']
        if term_sv not in inserted_term_sv:
            try:
                cursor.execute("""
                    INSERT INTO domain_embeddings (term, definition, topic, embedding)
                    VALUES (?, ?, ?, ?)
                """, (term_sv, term_info['def_sv'], term_info['topic'], term_info['embedding']))
                inserted_term_sv.add(term_sv)
                inserted_count += 1
            except sqlite3.IntegrityError:
                # Term already exists, likely from a previous run or different topic association
                # print(f"Info: Term '{term_sv}' already exists. Skipping duplicate insertion.")
                inserted_term_sv.add(term_sv) # Mark as handled for this run
                skipped_count += 1
            except sqlite3.Error as e:
                print(f"Error inserting term '{term_sv}': {e}")
                skipped_count += 1
        else:
            # Term already inserted in *this run* (via another topic). Skip.
            skipped_count += 1

    conn.commit()
    conn.close()
    print(f"Term loading complete. Inserted: {inserted_count}, Skipped (already exist or duplicate topic link): {skipped_count}")
