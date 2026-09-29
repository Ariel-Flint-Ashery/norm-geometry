
#%%

# a function that gets a word embedding from a language model given a specific token sequence and location of the token in the sequence.

# function that generates word embeddings for all words in a vocabulary using a language model




# function that gets a word embedding for an action label given all possible memories and the context of the game:
# generate prompt
# tokenize prompt
# tokenize word
# concatenate tokens
# get embeddings for last token


import yaml
from munch import munchify
import pickle
from _policy_router import get_prompt_options_embedding, get_initial_prompt_options_embedding
from utils import find_matrix_file
import torch.nn.functional as F

with open("generation_config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)


options_set = config.params.options_set
memory_size = config.params.memory_size_set[0]

try:
    options_set_fname = f"{options_set}_pairs.pkl"
    all_options = pickle.load(open(options_set_fname, 'rb'))
except FileNotFoundError:
    raise NotImplementedError("Only 'default' options_set is implemented.")


# options_id = list(options_dict.keys())[0]
# options = options_dict[options_id]['differences']
for options in all_options:
    print(options)
    # embeddings_dict = get_prompt_options_embedding(options = options, prompt = None)
    all_ordering_dict = get_initial_prompt_options_embedding(options = options)
    # print(all_ordering_dict.keys(), flush=True )
    similarities = []
    for options_string, embeddings_dict in all_ordering_dict.items():
        emb1, emb2 = embeddings_dict[options[0]], embeddings_dict[options[1]]
        cos_sim = F.cosine_similarity(emb1, emb2, dim=0)
        similarities.append(cos_sim.item())
    # print(similarities, flush=True)
    # get average distance
    avg_cos_sim = sum(similarities) / len(similarities)
    avg_cos_dist = 1 - avg_cos_sim

    output_dict = {'average_cosine_distance': avg_cos_dist, 'per_order_embeddings': {options_string: {'cosine_distance': 1 - sim, 'embeddings': {opt: all_ordering_dict[options_string][opt] for opt in options}} for options_string, sim in zip(all_ordering_dict.keys(), similarities)}}
    print(f'Options: {options}, Average Cosine distance: {avg_cos_dist}, single distances: {[1-s for s in similarities]}', flush=True)
    # save embeddings
    filename_pattern  = f"embeddings/initial_prompt_{config.model.shorthand}" + "_{name1}_{name2}_" + f"{config.params.temperature}tmp.pkl"
    matrix_fname = find_matrix_file(options, filename_pattern)
    with open(matrix_fname, 'wb') as f:
        pickle.dump(output_dict, f)

    embeddings_dict = get_prompt_options_embedding(options = options, prompt = None)
    emb1, emb2 = embeddings_dict[options[0]], embeddings_dict[options[1]]
    cos_sim = F.cosine_similarity(emb1, emb2, dim=0)
    cos_dist = 1 - cos_sim.item()
    output_dict = {'cosine_distance': cos_dist, 'embeddings': {opt: embeddings_dict[opt] for opt in options}}
    filename_pattern  = f"embeddings/word_only_{config.model.shorthand}" + "_{name1}_{name2}_" + f"{config.params.temperature}tmp.pkl"
    matrix_fname = find_matrix_file(options, filename_pattern)
    with open(matrix_fname, 'wb') as f:
        pickle.dump(output_dict, f)

