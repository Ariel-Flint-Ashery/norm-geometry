#%%
import numpy as np
import random
import pickle
from tqdm import tqdm
import multiprocessing as mp
import os
import shutil
from itertools import permutations
from scipy.special import softmax
import mean_field_module as mfm 

#%%
# In this experiment, we run a critical mass for a duration t < t_c.
# t_c is defined as the time taken for the critical mass committed minority to flip the population to the alternative steady state.
def get_critical_mass_data(data_dict):
    critical_mass_transition_dict = {
        s_idx: {
            committment_signal: (
                {
                    'critical_mass': data_dict[s_idx][committment_signal]['solution']['critical_mass'],
                    'time': data_dict[s_idx][committment_signal]['solution']['time']
                }
                if 'solution' in data_dict[s_idx][committment_signal]
                else None
            )
            for committment_signal in data_dict[s_idx].keys()
        }
        for s_idx in data_dict.keys()
    }

    return critical_mass_transition_dict

def init_duration_entry():
    return {
        'phase': 'commit',           # 'commit' | 'done'
        't_commit_done': 0,
        't_post_done': 0,
        't_total': 0,
        'early_stopping_time': None,
        'current_population': None,
        'short_action_history': np.full((3, 2), -np.inf),
        'flipped_during_commit': False,
        'flipped_during_commit_time': None,
        'long_action_history': []
    }

# def run_committed_phase(
#     entry, T_commit,
#     q_s, P_key, P_value, empty_state,
#     choices, nn_choices,
#     steady_state, committment_signal, c
# ):
#     llm = mfm.LLM_dynamics_integer(
#         q_s=q_s,
#         P_key=P_key,
#         P_value=P_value,
#         empty_state=empty_state,
#         choices=choices,
#         nn_choices=nn_choices,
#         steady_state=steady_state,
#         CM=c,
#         committment_signal=committment_signal,
#         population=entry['current_population']
#     )

#     if entry['current_population'] is None:
#         llm.initialize_population_steady()

#     actions_tracker = entry['long_action_history'].tolist()[-3:] if isinstance(entry['long_action_history'], np.ndarray) else entry['long_action_history'][-3:]
#     long_action_history = []
#     idx = 0  # index for circular buffer in short_action_history
#     while entry['t_commit_done'] < T_commit:
#         llm.algorithmic_update()
#         entry['t_commit_done'] += 1
#         action_probs = llm.population_log_probs
#         actions_tracker.append(action_probs)
#         actions_tracker = actions_tracker[-3:]
#         long_action_history.append(action_probs)
#         if len(actions_tracker) == 3:
#             avg_prob = np.mean(
#                 np.exp(np.array(actions_tracker)[:, committment_signal])
#             )
#             if avg_prob >= 0.98:
#                 entry['flipped_during_commit'] = True
#                 entry['flipped_during_commit_time'] = entry['t_commit_done']
        
#             # Overwrite oldest point
#             entry['short_action_history'][idx] = action_probs
#             idx = (idx + 1) % 3  # circular index
#     entry['current_population'] = llm.population
#     entry['phase'] = 'post'
#     if isinstance(entry['long_action_history'], np.ndarray):
#         entry['long_action_history'] = entry['long_action_history'].tolist()
    
#     short_action_history = long_action_history[-3:] + actions_tracker
#     entry['short_action_history'] = np.array(short_action_history[-3:])
#     entry['long_action_history'].extend(long_action_history)

def run_committed_phase(
    entry, T_commit,
    q_s, P_key, P_value, empty_state,
    choices, nn_choices,
    steady_state, committment_signal, c
):
    llm = mfm.LLM_dynamics_integer(
        q_s=q_s,
        P_key=P_key,
        P_value=P_value,
        empty_state=empty_state,
        choices=choices,
        nn_choices=nn_choices,
        steady_state=steady_state,
        CM=c,
        committment_signal=committment_signal,
        population=entry['current_population']
    )

    if entry['current_population'] is None:
        llm.initialize_population_steady()

    # Convert numpy array to list if loaded from saved file
    if isinstance(entry['long_action_history'], np.ndarray):
        entry['long_action_history'] = entry['long_action_history'].tolist()
    
    # Start circular buffer index where we left off
    # Calculate from the number of non-inf rows
    idx = np.sum(~np.isinf(entry['short_action_history'][:, 0])) % 3
    
    while entry['t_commit_done'] < T_commit:
        llm.algorithmic_update()
        entry['t_commit_done'] += 1
        entry['t_total'] += 1
        action_probs = llm.population_log_probs
        
        # Update circular buffer
        entry['short_action_history'][idx] = action_probs
        idx = (idx + 1) % 3
        
        # Append to long history
        entry['long_action_history'].append(action_probs)
        
        # Check flip condition only if we have 3 valid entries
        if entry['t_commit_done'] >= 3 and not entry['flipped_during_commit']:
            # Use circular buffer directly - get last 3 in order
            recent_3 = np.roll(entry['short_action_history'], -idx, axis=0)
            avg_prob = np.mean(np.exp(recent_3[:, committment_signal]))
            
            if avg_prob >= 0.98:
                entry['flipped_during_commit'] = True
                entry['flipped_during_commit_time'] = entry['t_commit_done']
    
    entry['current_population'] = llm.population
    entry['phase'] = 'done'

def run_simulation(
    fname,
    data_dict,
    q_total,
    cm_dict,
    H=3,
    commit_resolution=0.05,
    time_resolution=0.05,
    TIME=1000
):
    print("Precomputing reduced-state quantities...")
    q_H, keys_H, steady_0, steady_1 = mfm.H_reducer(q_total, H)
    F, Finv = mfm.integer_mapping(keys_H)
    choices, nn_choices = mfm.reverse_shift_vectorized(Finv)
    q_s = mfm.integer_probabilities(q_H, F)

    empty_state = F['']
    Ss = [F[steady_0], F[steady_1]]
    P_key, P_value = mfm.state_transitions(q_H, H, F, Finv)

    # initialize container if empty
    if not data_dict:
        data_dict = {
            s: {a: {} for a in range(len(Ss)) if a != s}
            for s in range(len(Ss))
        }

    print("Beginning critical mass evolution experiments...")

    for s_idx, steady_state in enumerate(Ss):
        for committment_signal in range(len(Ss)):
            if committment_signal == s_idx:
                continue

            cm_info = cm_dict[s_idx][committment_signal]
            if cm_info is None:
                print(f"No valid critical mass found for initial state {s_idx}, committed signal {committment_signal}. Skipping.")
                continue
            else:
                print(f"Loaded critical mass data for initial state {s_idx}, committed signal {committment_signal}: {cm_info}")
            T_c = cm_info['time']
            c = cm_info['critical_mass']
            if T_c is None or c is None:
                # print(f"No valid critical mass/time found for initial state {s_idx}, committed signal {committment_signal}. Skipping.")
                # continue
                c = 0.99
                T_c = TIME

            data_dict.setdefault(s_idx, {}).setdefault(committment_signal, {})

            # # Only one entry per (s_idx, committment_signal)
            # if s_idx not in data_dict:
            #     data_dict[s_idx] = {}
            # if committment_signal not in data_dict[s_idx]:
            #     data_dict[s_idx][committment_signal] = {}

            entry = data_dict[s_idx][committment_signal].get(c)
            if entry is None:
                entry = init_duration_entry()
                data_dict[s_idx][committment_signal][c] = entry
            if entry.get('phase') == 'done':
                print(f"Skipping completed run for initial state {s_idx}, committed signal {committment_signal}, critical mass {c}.")
                continue
            if entry.get('phase') == 'commit':

                # Run committed phase for T_c steps with critical mass c
                run_committed_phase(
                    entry, int(T_c),
                    q_s, P_key, P_value, empty_state,
                    choices, nn_choices,
                    steady_state, committment_signal, c
                )

                # Save after each run
                tmp = f"{fname}.tmp_{os.getpid()}"
                with open(tmp, "wb") as f:
                    pickle.dump(data_dict, f)
                shutil.move(tmp, fname)

    # Final checkpoint
    tmp = f"{fname}.tmp_{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(data_dict, f)
    shutil.move(tmp, fname)
    print("All critical mass evolution experiments complete.")


#%% LOAD q_total FROM FILES
def load_dataframe(fname):
    try:
        return pickle.load(open(fname, 'rb'))
    except:
        raise ValueError('NO DATAFILE FOUND')
    
def find_matrix_file(names, filename_pattern):
    for name1, name2 in permutations(names, 2):
        filename = filename_pattern.format(name1=name1, name2=name2)
        if os.path.exists(filename):
            return filename
    # If none found, return the one matching the input order
    name1, name2 = names
    return filename_pattern.format(name1=name1, name2=name2)

def run_single_option(args):
    """
    Function to run simulation for a single options_id.
    This function will be called by each process.
    """
    options, options_id, shorthand, H = args
    process_seed = os.getpid() + options_id  # Unique seed per process, random number
    random.seed(process_seed)
    np.random.seed(process_seed)
    # try:
    output_fname_pattern = f"policies/log_q_dicts/Q_dict_{shorthand}" +"_{name1}_{name2}_" + f"0.5tmp.pkl"
    q_dict_fname = find_matrix_file(options, output_fname_pattern)
    
    print(f"Process {os.getpid()}: Processing options pair {options}")
    print(f"Options: {options}")
    print("\nOption mapping:")
    for i, option in enumerate(options):
        print(f"  '{option}' -> {i}")
    
    with open(q_dict_fname, 'rb') as f:
        q_total = pickle.load(f)
    
    # load critical mass data (find CM and critical time)
    cm_file_pattern = f"meta_data/mean_field_critical_mass/LLM_dynamics_{shorthand}"+"_{name1}_{name2}_" + f"{H}mem_0.5tmp.pkl"
    cm_file = find_matrix_file(options, cm_file_pattern)
    
    try:
        with open(cm_file, 'rb') as f:
            cm_data_dict = pickle.load(f)
    except:
        raise ValueError('NO CRITICAL MASS DATAFILE FOUND')
    # generate cm_dict
    cm_dict = get_critical_mass_data(cm_data_dict)
    
    # remove cm_data_dict to save memory
    # del cm_data_dict

    # load experiment 1 data file
    output_file_pattern = f"meta_data/mean_field_committed/LLM_dynamics_{shorthand}"+"_{name1}_{name2}_" + f"{H}mem_0.5tmp.pkl"
    output_file = find_matrix_file(options, output_file_pattern)
    
    try:
        with open(output_file, 'rb') as f:
            data_dict = pickle.load(f)
    except:
        data_dict = {}

    print("Previously saved data points:")
    print(data_dict.keys())

    run_simulation(output_file, data_dict, q_total, cm_dict, H=H, TIME=1000)

    return f"Successfully completed options pair {options}"
        
    # except Exception as e:
    #     return f"Error processing options pair {options}: {str(e)}"

def run_parallel_simulations(max_processes=None):
    """
    Run simulations in parallel for different options_id values.
    
    Parameters:
    max_processes: Maximum number of processes to use. If None, uses all available CPU cores.
    """
    options_set_fname = "policies/emotion_pairs.pkl"
    all_options = pickle.load(open(options_set_fname, 'rb'))
    # all_options = [all_options[0]]
    shorthand = "phi_4" #"dsR1_Q32B" #"qwen25_7B" #"llama31_8B" #
    H = 3

    # Determine number of processes to use
    if max_processes is None:
        max_processes = mp.cpu_count()-1  # Leave one core free
    
    # Limit processes to available options or CPU cores, whichever is smaller
    num_processes = min(max_processes, len(all_options))
    
    print(f"Running simulations on {num_processes} processes for {len(all_options)} options")
    print(f"Available CPU cores: {mp.cpu_count()}")
    
    # Prepare arguments for each process
    process_args = [(options, options_id, shorthand, H) for options_id, options in enumerate(all_options)]
    
    # Create and start processes
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(run_single_option, process_args)
    
    # Print results
    print("\n" + "="*50)
    print("SIMULATION RESULTS:")
    print("="*50)
    for result in results:
        print(result)

if __name__ == "__main__":
    # You can specify the maximum number of processes to use
    # If you want to use all available CPU cores, use None
    # If you want to limit the number of processes, specify a number
    run_parallel_simulations(max_processes=10)
    
# %%
