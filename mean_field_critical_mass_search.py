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
# Revised critical-mass search version of run_simulation()
# Includes monotonic (binary-style) search over c, with grid snapping to 0.01
# Drop-in compatible with existing code structure

# def snap_to_grid(c, resolution=0.01):
#     return round(round(c / resolution) * resolution, 2)

def run_committed_phase(data, s_idx, steady_state, committment_signal, c, q_s, P_key, P_value, empty_state, choices, nn_choices, TIME=100, euler_step=0.1):
    # -----------------------------
    # RUN THE SIMULATIONS
    # -----------------------------
    print(f"----Starting simulation for steady state {s_idx}, committed signal {committment_signal}, c={c:.2f}----")
    llm = mfm.LLM_dynamics_integer(
        q_s=q_s, P_key=P_key, P_value=P_value, empty_state=empty_state, choices=choices, nn_choices=nn_choices,
        steady_state=steady_state, CM=c, committment_signal=committment_signal, population = data['current_population'], dt = euler_step
    )
    if data['current_population'] is None:
        # print("Initializing population to steady state...")
        llm.initialize_population_steady()

    actions_tracker = []

    for t in tqdm(range(TIME)):
        llm.algorithmic_update_asynchronous()
        action_probs = llm.population_log_probs
        # do convergence checks

        if len(actions_tracker) < 3:
            actions_tracker.append(action_probs)
        else:
            actions_tracker.pop(0)
            actions_tracker.append(action_probs)

            # Convergence
            x = np.array(actions_tracker)
            avg_action_prob = np.mean(np.exp(x[:, committment_signal]))

            if avg_action_prob >= 0.98:
                data['early_stopping_time'] = t + 1
                data['current_population'] = llm.population
                data['short_action_history'] = x
                data['total_timesteps'] += t + 1
                break

        # TIME exhausted
        if t == TIME - 1:
            data['current_population'] = llm.population
            data['short_action_history'] = np.array(actions_tracker)
            data['total_timesteps'] += t + 1

def run_simulation(fname, data_dict, q_total, H=3, TIME=100, resolution = 0.01, euler_step=0.1):
    # Precompute reduced-state quantities
    print("Precomputing reduced-state quantities...")
    q_H, keys_H, steady_0, steady_1 = mfm.H_reducer(q_total, H)
    F, Finv = mfm.integer_mapping(keys_H)
    choices, nn_choices = mfm.reverse_shift_vectorized(Finv)
    print(f"Integer mapping complete. Total reduced states: {len(F)}")
    q_s = mfm.integer_probabilities(q_H, F)
    empty_state = F['']
    Ss = [F[steady_0], F[steady_1]]
    # print(f"Steady states integer-mapped: {Ss}")
    P_key, P_value = mfm.state_transitions(q_H, H, F, Finv)

    # calculate real timesteps based on euler_step
    # TIME = int(TIME / euler_step)

    if len(data_dict.keys()) == 0:
        data_dict = {i: {j: {} for j in range(len(Ss)) if j != i} for i in range(len(Ss))}

    # Critical mass transition matrices (not required for algorithmic_update path)

    print("Setup complete. Beginning simulations...")
    for s_idx, steady_state in enumerate(Ss):
        alternative_indices = [i for i in range(len(Ss)) if i != s_idx]
        for committment_signal in alternative_indices:
            # -------------------------------------------------
            # SAFE DISCRETE BINARY SEARCH OVER c
            # -------------------------------------------------
            
            # create a discrete grid from 0 to 1 with step size resolution
            grid = [round(i * resolution, 2) for i in range(int(1/resolution) + 1)]

            lo = 0
            hi = len(grid) - 1

            while lo <= hi:

                mid = (lo + hi) // 2
                c = grid[mid]

                # ---------------------------------------
                # INITIALISE / RESUME DATA ENTRY
                # ---------------------------------------
                if c not in data_dict[s_idx][committment_signal]:
                    print(f"Creating new entry for Steady={s_idx}, c={c:.2f}")
                    data_dict[s_idx][committment_signal][c] = {
                        'early_stopping_time': None,
                        'current_population': None,
                        'short_action_history': np.full((3, 2), -np.inf),
                        # 'action_probabilities_history': [],
                        # 'population_history': [],
                        'total_timesteps': 0,
                    }
                    TIME_TO_RUN = TIME
                else:
                    prev = data_dict[s_idx][committment_signal][c]['total_timesteps']
                    TIME_TO_RUN = TIME - prev
                    if TIME_TO_RUN <= 0 or data_dict[s_idx][committment_signal][c]['early_stopping_time'] is not None:
                        print(f"Skipping completed s={s_idx}, A={committment_signal}, c={c:.2f}")

                        alt_log_production_p_history = data_dict[s_idx][committment_signal][c]['short_action_history'][:, committment_signal]
                        # get average of history
                        avg_action_prob = np.mean(np.exp(alt_log_production_p_history))
                        if avg_action_prob >= 0.98:
                            # always wins → threshold lower
                            hi = mid - 1
                        else:
                            # sometimes fails → threshold higher
                            lo = mid + 1
                        continue
                    print(f"Resuming s={s_idx}, A={committment_signal}, c={c:.2f}. Already ran {prev} iterations.")

                data = data_dict[s_idx][committment_signal][c]

                # RUN SIMULATION PHASE
                run_committed_phase(
                    data, s_idx, steady_state, committment_signal, c, q_s, P_key, P_value, empty_state, choices, nn_choices,
                    TIME=TIME_TO_RUN, euler_step=euler_step
                )

                # Safe save
                temp_fname = f"{fname}.tmp_{os.getpid()}"
                with open(temp_fname, "wb") as f:
                    pickle.dump(data_dict, f)
                shutil.move(temp_fname, fname)

                # ---------------------------------------------------
                # DECISION: option A monotonic interpretation
                # ---------------------------------------------------
                # If committed minority ALWAYS wins → threshold is LOWER → search left half
                final_action_probs_history = data['short_action_history'][:, committment_signal]
                # print("Final action probabilities history (committed signal):", np.exp(final_action_probs_history))
                average_final_action_prob = np.mean(np.exp(final_action_probs_history))
                # print("Average final action probability (committed signal):", average_final_action_prob)
                if average_final_action_prob >= 0.98:
                    print(f"All runs flipped at c={c:.2f}. Searching LOWER c.")
                    hi = mid - 1
                else:
                    print(f"Not all runs flipped at c={c:.2f}. Searching HIGHER c.")
                    lo = mid + 1
            # End of discrete binary search
            data = data_dict[s_idx][committment_signal][c]
            final_action_probs_history = data['short_action_history'][:, committment_signal]
            average_final_action_prob = np.mean(np.exp(final_action_probs_history))
            if lo >= len(grid):
                print(f"----Critical mass COMPLETE. <<<SEARCH FAILED>>> for initial state {s_idx}, committed signal {committment_signal}.----")
                fin_strat = np.mean(np.exp(data['short_action_history']), axis = 0)
                print(f"Final c tried: {c:.2f} with final strategy probabilities: {fin_strat}")
                data_dict[s_idx][committment_signal]['solution'] = {'critical_mass': None, 'time': None}
            else:
                print(f"----Critical mass COMPLETE. All simulations converged to steady state {committment_signal} from initial state {s_idx}, c={c:.2f} after {data['total_timesteps']} timesteps.----")
                data_dict[s_idx][committment_signal]['solution'] = {'critical_mass': c, 'time': data['total_timesteps']}

                # search for true critical mass (the smallest c that always flips)
                
                # first, check if c is already the smallest grid value that flips
                if average_final_action_prob >=0.98 or c == 0.0:
                    print("c is already the smallest grid value that flips.")
                    continue

                # check if we already have the smallest grid value that flips
                sorted_cm_values = sorted([key for key in data_dict[s_idx][committment_signal].keys() if isinstance(key, float)])
                c_index = sorted_cm_values.index(c)
                if abs(c - sorted_cm_values[c_index+1]) <= resolution:
                    print("Already have the smallest grid value that flips.")
                    # record smallest grid value that flips as critical mass
                    new_c = sorted_cm_values[c_index+1]
                    data = data_dict[s_idx][committment_signal][new_c]
                    data_dict[s_idx][committment_signal]['solution'] = {'critical_mass': new_c, 'time': data['total_timesteps']}
                    continue

                # otherwise, search upwards to find the true critical mass
                true_cm = c + resolution
                assert true_cm in grid, "True critical mass not in grid!"
                # run simulations at true_cm if not already done
                data_dict[s_idx][committment_signal][true_cm] = {
                        'early_stopping_time': None,
                        'current_population': None,
                    'short_action_history': np.full((3, 2), -np.inf),
                        # 'action_probabilities_history': [],
                        # 'population_history': [],
                        'total_timesteps': 0,
                    }
                data = data_dict[s_idx][committment_signal][true_cm]
                run_committed_phase(data, s_idx, steady_state, committment_signal, true_cm, q_s, P_key, P_value, empty_state, choices, nn_choices, TIME=TIME)
                # Safe save
                temp_fname = f"{fname}.tmp_{os.getpid()}"
                with open(temp_fname, "wb") as f:
                    pickle.dump(data_dict, f)
                shutil.move(temp_fname, fname)

                # If committed minority ALWAYS wins → threshold is LOWER → search left half
                final_action_probs_history = data['short_action_history'][:, committment_signal]
                # print("Final action probabilities history (committed signal):", np.exp(final_action_probs_history))
                average_final_action_prob = np.mean(np.exp(final_action_probs_history))
                print(f"----Critical mass COMPLETE <<<TRUE VALUE>>>. All simulations converged to steady state {committment_signal} from initial state {s_idx}, c={true_cm:.2f} after {data['total_timesteps']} timesteps.----")
                data_dict[s_idx][committment_signal]['solution'] = {'critical_mass': true_cm, 'time': data['total_timesteps']}

            # safe save at the end of each committed signal
            temp_fname = f"{fname}.tmp_{os.getpid()}"
            with open(temp_fname, "wb") as f:
                pickle.dump(data_dict, f)
            shutil.move(temp_fname, fname)
    # safe save at the end of each committed signal
    temp_fname = f"{fname}.tmp_{os.getpid()}"
    with open(temp_fname, "wb") as f:
        pickle.dump(data_dict, f)
    shutil.move(temp_fname, fname)

#%% LOAD q_total FROM FILES
def load_dataframe(fname):
    try:
        return pickle.load(open(fname, 'rb'))
    except:
        raise ValueError('NO DATAFILE FOUND')
    
def find_matrix_file(names, filename_pattern):
    # Support variable-length name tuples; here we expect pairs
    for perm in permutations(names, len(names)):
        if len(perm) == 2:
            name1, name2 = perm
            filename = filename_pattern.format(name1=name1, name2=name2)
        else:
            # Fallback: try to format whatever placeholders exist
            try:
                filename = filename_pattern.format(*perm)
            except Exception:
                continue
        if os.path.exists(filename):
            return filename
    # If none found, return the one matching the input order (assumes two names)
    if len(names) == 2:
        name1, name2 = names
        return filename_pattern.format(name1=name1, name2=name2)
    # As last resort return pattern untouched (likely to fail fast elsewhere)
    return filename_pattern

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
    # Use 2-option q_dicts from policies/q_dicts
    output_fname_pattern = f"policies/log_q_dicts/Q_dict_{shorthand}" + "_{name1}_{name2}_" + f"0.5tmp.pkl"
    q_dict_fname = find_matrix_file(options, output_fname_pattern)
    
    print(f"Process {os.getpid()}: Processing options pair {options}")
    print(f"Options: {options}")
    print("\nOption mapping:")
    for i, option in enumerate(options):
        print(f"  '{option}' -> {i}")
    
    with open(q_dict_fname, 'rb') as f:
        q_total = pickle.load(f)
    
    output_file_pattern = f"meta_data_async/mean_field_critical_mass/LLM_dynamics_{shorthand}" + "_{name1}_{name2}_" + f"{H}mem_0.5tmp.pkl"
    output_file = find_matrix_file(options, output_file_pattern)
    
    try:
        with open(output_file, 'rb') as f:
            data_dict = pickle.load(f)
    except:
        data_dict = {}

    print("Previously saved data points:")
    print(data_dict.keys())

    run_simulation(output_file, data_dict, q_total, H=H, TIME=1000)

    return f"Successfully completed options pair {options}"
        
    # except Exception as e:
    #     return f"Error processing options pair {options}: {str(e)}"

def run_parallel_simulations(max_processes=None):
    """
    Run simulations in parallel for different options_id values.
    
    Parameters:
    max_processes: Maximum number of processes to use. If None, uses all available CPU cores.
    """
    # Load 2-option sets
    options_set_fname = "policies/letter_pairs.pkl"
    all_options = pickle.load(open(options_set_fname, 'rb'))
    # all_options = [all_options[0]]  # uncomment to test a single pair
    shorthand = "qwen25_7B" #"phi_4" #"dsR1_Q32B" #"llama32_3B" # "gpt-oss-20b" #
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
    run_parallel_simulations(max_processes=4)
    
# %%
