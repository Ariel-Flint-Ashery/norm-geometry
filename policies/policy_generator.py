
#%%
import yaml
from munch import munchify
import pickle
from _policy_router import get_full_transition_matrix


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
    get_full_transition_matrix(options = options, memory_size=memory_size)