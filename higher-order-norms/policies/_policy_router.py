#%%
import sqlite3
import json
from itertools import permutations
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from munch import munchify
import yaml
import pickle
import prompting_module as pr
import utils as ut

#%%
with open("generation_config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)

# if config.sim.mode == 'api':
#     import api_connection as ask
if config.sim.mode == 'gpu':
    import gpu_connection as ask

_connection_cache = {}

def get_db_connection(fname):
    """
    Get a persistent connection to the database, creating it if necessary.
    """
    global _connection_cache
    
    if fname not in _connection_cache or _connection_cache[fname] is None:
        # Create a new connection
        conn = sqlite3.connect(fname, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")  # Enable concurrent reads while writing
        conn.execute("PRAGMA synchronous=NORMAL;")  # Reduce disk I/O
        conn.execute("PRAGMA cache_size=10000;")   # Increase cache size
        
        # Create table if it doesn't exist
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transition_matrix (
                memory_string TEXT,
                options_string TEXT,
                probability_dict TEXT,
                PRIMARY KEY (memory_string, options_string)
            )
        """)
        conn.commit()
        
        # Store in cache
        _connection_cache[fname] = conn
    
    return _connection_cache[fname]

def close_all_connections():
    """
    Close all database connections.
    Should be called at the end of the program.
    """
    global _connection_cache
    
    for fname, conn in _connection_cache.items():
        if conn is not None:
            try:
                conn.close()
            except Exception as e:
                print(f"Error closing connection to {fname}: {e}")
    
    _connection_cache = {}

def get_transition_matrix(fname, options, my_history, partner_history, prompt, first_target_id_dict):
    """
    Loads or computes a transition probability dictionary using a persistent connection.
    """
    # Create memory key for caching/lookup (preserving original data structure)
    memory_string = '_'.join(my_history) + '_&_' + '_'.join(partner_history)
    options_string = '_'.join(options)

    # Get persistent connection
    conn = get_db_connection(fname)
    cursor = conn.cursor()
    
    try:
        # Check if the entry exists
        cursor.execute("SELECT probability_dict FROM transition_matrix WHERE memory_string = ? AND options_string = ?", 
                    (memory_string, options_string))
        row = cursor.fetchone()

        if row:
            # Load existing probability_dict
            probability_dict = json.loads(row[0])
        else:
            # Compute new transition probabilities (preserving original parameters)
            probability_dict = ask.get_probability_dict(options=options, prompt=prompt, first_target_id_dict=first_target_id_dict)

            # Save to database using a transaction for atomicity
            cursor.execute("INSERT OR REPLACE INTO transition_matrix (memory_string, options_string, probability_dict) VALUES (?, ?, ?)", 
                        (memory_string, options_string, json.dumps(probability_dict)))
            conn.commit()
    except sqlite3.Error as e:
        # Handle potential database errors
        print(f"Database error: {e}")
        conn.rollback()
        raise
        
    return probability_dict


def get_full_transition_matrix(options, memory_size):
    first_target_id_dict = ask.encode_decode_options(options = options)
    # get empty memory transitions
    action_vectors = ut.generate_action_vectors(options = options, memory_size = memory_size)
    all_options = list(permutations(options))
    for opts in all_options:
        opts = list(opts)
        #print(opts)
        for pair in tqdm(action_vectors):
            #print(pair)
            player=ut.get_player()
            m = len(pair[0])
            for h in range(m):
                my_answer, partner_answer = [p[h] for p in pair]
                outcome = ut.get_outcome(my_answer, partner_answer)
                ut.update_dict(player, my_answer, partner_answer, outcome)

            rules = pr.get_rules(options = opts)
            # get prompt with rules & history of play
            prompt = pr.get_prompt(player = player, memory_size=m, rules = rules)
            # get agent response
            filename_pattern  = f"matrices/TRANSITION_MATRIX_{config.model.shorthand}" + "_{name1}_{name2}_{name3}_" + f"{config.params.temperature}tmp.db"
            matrix_fname = ut.find_matrix_file(options, filename_pattern)
            matrix = get_transition_matrix(fname = matrix_fname, options = opts, my_history=player['my_history'], partner_history=player['partner_history'], prompt = prompt, first_target_id_dict=first_target_id_dict)

def get_prompt_options_embedding(prompt, options):
    first_target_id_dict = ask.encode_decode_options(options = options)
    embeddings = ask.get_word_embedding_from_prompt(text = prompt, first_target_id_dict=first_target_id_dict)
    return embeddings

def get_initial_prompt_options_embedding(options):
    all_options = list(permutations(options))
    output_dict = {}
    for opts in all_options:
        opts = list(opts)
        player=ut.get_player()
        m = 0
        rules = pr.get_rules(options = opts)
        # get prompt with rules & history of play
        prompt = pr.get_prompt(player = player, memory_size=m, rules = rules)
        # print(prompt, flush=True)
        embeddings = get_prompt_options_embedding(prompt = prompt, options = opts)
        options_string = '_'.join(opts)
        output_dict[options_string] = embeddings
    return output_dict

    





