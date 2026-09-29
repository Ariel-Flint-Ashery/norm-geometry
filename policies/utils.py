from scipy.special import logsumexp
import random
import numpy as np
from itertools import product, permutations
import os

def normalize_logprobs(logprobs):
    logtotal = logsumexp(logprobs) #calculates the summed log probabilities
    normedlogs = []
    for logp in logprobs:
        normedlogs.append(logp - logtotal) #normalise - subtracting in the log domain equivalent to divising in the normal domain
    return normedlogs

def normalize_probs(probs):
    total = sum(probs) #calculates the summed probabilities
    normedprobs = []
    for p in probs:
        normedprobs.append(p / total) 
    return normedprobs

def roulette_wheel(normedprobs):
    r=random.random() #generate a random number between 0 and 1
    accumulator = normedprobs[0]
    for i in range(len(normedprobs)):
        if r < accumulator:
            return i
        accumulator = accumulator + normedprobs[i + 1]

def log_roulette_wheel(logprobs):
    return np.argmax(np.array(logprobs) + np.random.gumbel(size = len(logprobs)))

def generate_action_vectors(memory_size=5, options=[0, 1]):
    all_vectors = []
    for rounds in range(memory_size + 1):
        player1_choices = product(options, repeat=rounds)
        player2_choices = product(options, repeat=rounds)
        for p1, p2 in product(player1_choices, player2_choices):
            all_vectors.append((list(p1), list(p2)))
    return all_vectors

def get_player():
    return {'my_history': [], 'partner_history': [], 'score': 0, 'outcome': []}

def get_outcome(my_answer, partner_answer, rewards = [-50, 100]):
    if my_answer == partner_answer:
        return rewards[1]
    return rewards[0]

def update_dict(player, my_answer, partner_answer, outcome):
  player['score'] += outcome
  player['my_history'].append(my_answer)
  player['partner_history'].append(partner_answer)
  player['outcome'].append(outcome)

  return player

def find_matrix_file(names, filename_pattern):
    for name1, name2 in permutations(names, 2):
        filename = filename_pattern.format(name1=name1, name2=name2)
        if os.path.exists(filename):
            return filename
    # If none found, return the one matching the input order
    name1, name2 = names
    return filename_pattern.format(name1=name1, name2=name2)
