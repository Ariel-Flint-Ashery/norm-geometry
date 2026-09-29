import numpy as np
import random
from numba import float64, int64
from numba.experimental import jitclass
import pickle
from tqdm import tqdm, trange
from copy import deepcopy
import multiprocessing as mp
import os
import shutil
from itertools import permutations

#%% BASICS
def shift(key, choice, nn_choice, H):
    """
        key grows until it reaches the length 2*H, then it shifts to the left

    """

    if len(key) < 2*H:
        return key + str(choice) + str(nn_choice)
    else:
        return key[2:] + str(choice) + str(nn_choice)

# Random mapping beteween states and integers

def integer_mapping(keys_H):
    """
        Create a random mapping between the states and integers
        Returns:
            F: dictionary mapping states to integers
            Finv: dictionary inverse mapping integers to states

    """        

    F = {} # direct function
    Finv = {} # inverse function

    item = 0

    for k in keys_H:
        F[k] = item
        Finv[item] = k
        item += 1

    return F, Finv

def state_transitions(q_H, H, F, Finv):
    """
        Compute the transition tensor P
        Returns:
            P_key: state of the transition
            P_value: probability of the transition
    """

    n = len(q_H)

    P_value = np.zeros((n, n, 4), dtype=float) # Store as a numpy array
    P_key = np.zeros((n, n, 4), dtype=int) # Store as a numpy array

    for i in range(n):
        for j in range(n):
            key_i = Finv[i]
            key_j = Finv[j]

            prob_0_i = q_H[key_i]
            prob_0_j = q_H[key_j]

            next_i_1 = F[shift(key_i, 0, 0, H)]
            next_i_2 = F[shift(key_i, 0, 1, H)]
            next_i_3 = F[shift(key_i, 1, 0, H)]
            next_i_4 = F[shift(key_i, 1, 1, H)]


            P_key[i,j,0] = next_i_1
            P_key[i,j,1] = next_i_2
            P_key[i,j,2] = next_i_3
            P_key[i,j,3] = next_i_4
            P_value[i,j,0] = prob_0_i*prob_0_j
            P_value[i,j,1] = prob_0_i*(1-prob_0_j)
            P_value[i,j,2] = (1-prob_0_i)*prob_0_j
            P_value[i,j,3] = (1-prob_0_i)*(1-prob_0_j)  

    return P_key, P_value

def integer_probabilities(q_H, F):
    """
        Compute the probability of output 0 for each state
        Returns:
            q_s: probability of output 0 for each state
    """
    n = len(q_H)

    q_s = np.zeros(n)
    for k in F:
        i = F[k]

        q_s[i] = q_H[k]

    return q_s

def H_reducer(q_total, H):
    """
        Reduce the dictionary to the keys of length 2*H
    """
    q_H = {k: v for k, v in q_total.items() if len(k) <= 2*H}

    keys_H = set(q_H.keys())

    steady_0 = ''.join(['0']*(2*H))
    steady_1 = ''.join(['1']*(2*H))

    return q_H, keys_H, steady_0, steady_1

spec = [
    ('N', int64),
    ('q_s', float64[:]),
    ('population', int64[:]),
    ('P_key', int64[:,:,:]),
    ('empty_state', int64),
    ('steady_state', int64),
    ('CM', int64),
    ('committment_signal', int64)
]

@jitclass(spec)
class LLM_dynamics_integer:
    """
        Simple class simulator
    """
    def __init__(self, N, q_s, P_key, empty_state, steady_state, CM, committment_signal):
        self.N = N
        self.q_s = q_s
        self.population = np.zeros(N, dtype=np.int64)
        self.P_key = P_key
        self.empty_state = empty_state
        self.CM = CM
        self.steady_state = steady_state
        self.committment_signal = committment_signal

        
    def initialize_population_zero(self):
        """
            Initialize the population of LLMs to the empty state
        """
        
        for i in range(self.N):
            self.population[i] = self.empty_state

    def initialize_population_random(self):
        """
            Initialize the population of LLMs
        """

        n  = len(self.q_s)
        
        for i in range(self.N):
            k = random.randint(0, n-1)
            self.population[i] = k
    
    def initialize_population_steady(self):
        """
            Initialize the population of LLMs to a steady state
        """

        for i in range(self.N):
            self.population[i] = self.steady_state

    def update(self):
        """
        Update the population of LLMs one Monte Carlo time step.
        Agents with index < CM are committed and always output '1' without updating state.
        """
        update_output_dict = {'all_words': 0, 'all_success': 0, 'all_count': 0}#,
                            #    'mixed_success': 0, 'mixed_words': 0, 'mixed_count': 0,
                            #      'uncommitted_success': 0, 'uncommitted_words': 0, 'uncommitted_count': 0}
        
        at_least_one_uncommitted = False
        while at_least_one_uncommitted==False:
            for s in range(self.N):
                # Choose two different LLMs
                i = random.randint(0, self.N - 1)
                j = random.randint(0, self.N - 1)
                while j == i:
                    j = random.randint(0, self.N - 1)

                # Determine if agents are committed
                i_committed = i < self.CM
                j_committed = j < self.CM

                # Generate outputs based on commitment status
                if i_committed:
                    first = self.committment_signal  # Committed agents always output the commitment signal
                else:
                    prob_i = self.q_s[self.population[i]]
                    rr = random.random()
                    first = 1 - int(rr < prob_i)

                if j_committed:
                    second = self.committment_signal  # Committed agents always output the commitment signal
                else:
                    prob_j = self.q_s[self.population[j]]
                    rr = random.random()
                    second = 1 - int(rr < prob_j)

                # Calculate interaction keys
                k_i = 2 * first + second
                k_j = 2 * second + first

                # Update states for all agents (committed and non-committed)
                temp_i = self.population[i]
                temp_j = self.population[j]
                self.population[i] = self.P_key[temp_i, temp_j, k_i]
                self.population[j] = self.P_key[temp_j, temp_i, k_j]

                success = int(first == second)
                # Track metrics (entire population)
                # update_output_dict['all_words'] += first + second
                # update_output_dict['all_success'] += success

                if int(i_committed)*int(j_committed) == 1:
                    continue  # both committed, skip
                

                all_words = first * (1 - int(i_committed)) + second * (1 - int(j_committed))
                # Track metrics (uncommitted is present)
                # if not i_committed or not j_committed:
                update_output_dict['all_count'] += (1-int(i_committed)) + (1-int(j_committed)) # count uncommitted agents
                update_output_dict['all_success'] += success
                update_output_dict['all_words'] += all_words #first * (1 - int(i_committed)) + second * (1 - int(j_committed))
                    
                
                # # Track metrics (uncommitted and committed)
                # if sum([int(i_committed), int(j_committed)]) == 1:
                #     update_output_dict['mixed_words'] += all_words
                #     update_output_dict['mixed_success'] += success
                #     update_output_dict['mixed_count'] += 1
                #     continue
                
                # # Track metrics (both uncommitted)
                # if not i_committed and not j_committed:
                #     update_output_dict['uncommitted_count'] += 2
                #     update_output_dict['uncommitted_words'] += all_words #first + second
                #     update_output_dict['uncommitted_success'] += success
                #     continue
            at_least_one_uncommitted = update_output_dict['all_count']>0
            
        # compute final metrics
        for prefix in ['all']:#, 'mixed', 'uncommitted']:
            count = update_output_dict[f'{prefix}_count']
            if count > 0:
                update_output_dict[f'{prefix}_success'] /= count
                update_output_dict[f'{prefix}_words'] /= count
            else:
                update_output_dict[f'{prefix}_success'] = None
                update_output_dict[f'{prefix}_words'] = None

        return update_output_dict
        # return success / self.N, all_words / (2 * self.N)
    
# Revised critical-mass search version of run_simulation()
# Includes monotonic (binary-style) search over c, with grid snapping to 0.01
# Drop-in compatible with existing code structure

def snap_to_grid(c, resolution=0.01):
    return round(round(c / resolution) * resolution, 2)


def run_simulation(fname, data_dict, q_total, Ns=None, H=5, ITERS=1000, TIME=1000):
    # Precompute reduced-state quantities
    q_H, keys_H, steady_0, steady_1 = H_reducer(q_total, H)
    F, Finv = integer_mapping(keys_H)
    q_s = integer_probabilities(q_H, F)
    empty_state = F['']
    Ss = [F[steady_0], F[steady_1]]
    P_key, P_value = state_transitions(q_H, H, F, Finv)

    if Ns is None:
        Ns = np.geomspace(10, 10000, 25, dtype=int)

    # User-provided Cs ignored for search; we use 0.00–1.00 grid implicitly

    Ns = sorted(Ns)

    for N in Ns:
        if N not in data_dict:
            data_dict[N] = {0: {}, 1: {}}

        for s_idx, steady_state in enumerate(Ss):
            # -------------------------------------------------
            # SAFE DISCRETE BINARY SEARCH OVER c
            # -------------------------------------------------
            committment_signal = 0 if steady_state == F[steady_1] else 1
            # Create discrete grid c ∈ {0.00, 0.01, ..., 1.00}
            grid = [round(i * 0.01, 2) for i in range(101)]

            lo = 0
            hi = len(grid) - 1

            while lo <= hi:

                mid = (lo + hi) // 2
                c = grid[mid]
                CM = int(round(N * c))

                # ---------------------------------------
                # INITIALISE / RESUME DATA ENTRY
                # ---------------------------------------
                if c not in data_dict[N][s_idx]:
                    print(f"Creating new entry for N={N}, Steady={s_idx}, c={c:.2f}")
                    data_dict[N][s_idx][c] = {
                        0: {'realisations': 0, 'time': []},
                        1: {'realisations': 0, 'time': []},
                        'mixed': {
                            0: {'realisations': 0, 'time': []},
                            1: {'realisations': 0, 'time': []}
                        },
                        'total_attempts': 0,
                        'CM_count': CM
                    }
                    ITERS_TO_RUN = ITERS
                else:
                    prev = data_dict[N][s_idx][c]['total_attempts']
                    ITERS_TO_RUN = ITERS - prev
                    if ITERS_TO_RUN <= 0:
                        print(f"Skipping completed N={N}, s={s_idx}, c={c:.2f}")
                        # # Move to next midpoint
                        CM_wins = data_dict[N][s_idx][c][committment_signal]['realisations']

                        if CM_wins >= ITERS:
                            # always wins → threshold lower
                            hi = mid - 1
                        else:
                            # sometimes fails → threshold higher
                            lo = mid + 1
                        continue
                    print(f"Resuming N={N}, s={s_idx}, c={c:.2f}. Already ran {prev} iterations.")

                data = data_dict[N][s_idx][c]

                # -----------------------------
                # RUN THE SIMULATIONS
                # -----------------------------
                for it in trange(ITERS_TO_RUN, desc=f"PID {os.getpid()} | N:{N} | c:{c:.2f}"):

                    llm = LLM_dynamics_integer(
                        N, q_s, P_key, empty_state,
                        steady_state, CM, committment_signal
                    )
                    llm.initialize_population_steady()

                    success_tracker = []
                    words_tracker = []

                    for t in range(TIME):
                        # success, words = llm.update()
                        update_output_dict = llm.update()
                        success = update_output_dict['all_success']
                        words = update_output_dict['all_words']

                        if len(success_tracker) < 3:
                            success_tracker.append(success)
                            words_tracker.append(words)
                        else:
                            success_tracker.pop(0)
                            words_tracker.pop(0)
                            success_tracker.append(success)
                            words_tracker.append(words)

                            # Convergence
                            # if np.mean(success_tracker) >= 0.98:
                            # check if commitmment signal has been adopted by uncommitted agents
                            words_mean = np.mean(words)
                            if (committment_signal == 1 and words_mean >= 0.98) or (committment_signal == 0 and words_mean <= 0.02):
                                final_state = int(words_mean >= 0.5)
                                data[final_state]['time'].append(t)
                                data[final_state]['realisations'] += 1
                                break


                        # TIME exhausted
                        if t == TIME - 1:
                            final_state = int(np.mean(words_tracker) >= 0.5)
                            data['mixed'][final_state]['time'].append(t)
                            data['mixed'][final_state]['realisations'] += 1

                # Update attempts
                data['total_attempts'] += ITERS_TO_RUN

                # Safe save
                temp_fname = f"{fname}.tmp_{os.getpid()}"
                with open(temp_fname, "wb") as f:
                    pickle.dump(data_dict, f)
                shutil.move(temp_fname, fname)

                num_0 = data[0]['realisations']
                num_1 = data[1]['realisations']

                # ---------------------------------------------------
                # DECISION: option A monotonic interpretation
                # ---------------------------------------------------
                # If committed minority ALWAYS wins → threshold is LOWER → search left half
                if data[committment_signal]['realisations'] >= ITERS:
                    print(f"All runs flipped at c={c:.2f}. Searching LOWER c.")
                    hi = mid - 1
                else:
                    print(f"Not all runs flipped at c={c:.2f}. Searching HIGHER c.")
                    lo = mid + 1
            print(f"----Critical mass COMPLETE. All simulations converged to steady state {committment_signal} for N={N}, c={c:.2f}.----")


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
    options, options_id, shorthand, H, Ns = args
    process_seed = os.getpid() + options_id  # Unique seed per process, random number
    random.seed(process_seed)
    np.random.seed(process_seed)
    try:
        output_fname_pattern = f"policies/q_dicts/Q_dict_{shorthand}" +"_{name1}_{name2}_" + f"0.5tmp.pkl"
        q_dict_fname = find_matrix_file(options, output_fname_pattern)
        
        print(f"Process {os.getpid()}: Processing options pair {options}")
        print(f"Options: {options}")
        print("\nOption mapping:")
        for i, option in enumerate(options):
            print(f"  '{option}' -> {i}")
        
        with open(q_dict_fname, 'rb') as f:
            q_total = pickle.load(f)
        
        output_file_pattern = f"meta_data/committed/LLM_dynamics_{shorthand}"+"_{name1}_{name2}_" + f"{H}mem_0.5tmp.pkl"
        output_file = find_matrix_file(options, output_file_pattern)
        
        try:
            with open(output_file, 'rb') as f:
                data_dict = pickle.load(f)
        except:
            data_dict = {}

        print("Previously saved data points:")
        print(data_dict.keys())

        run_simulation(output_file, data_dict, q_total, Ns=Ns, H=H, ITERS=500, TIME=1000)

        return f"Successfully completed options pair {options}"
        
    except Exception as e:
        return f"Error processing options pair {options}: {str(e)}"

def run_parallel_simulations(max_processes=None):
    """
    Run simulations in parallel for different options_id values.
    
    Parameters:
    max_processes: Maximum number of processes to use. If None, uses all available CPU cores.
    """
    options_set_fname = "policies/emotion_pairs.pkl"
    all_options = pickle.load(open(options_set_fname, 'rb'))
    shorthand = "dsR1_Q32B" # #"llama31_8B"#"qwen25_7B" # "phi_4" #
    H = 3
    Ns = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,5000, 10000]
    # get number of CM for each N (rounding)
    # CM_fractions =  [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 
    #                  0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,
    #                  0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3,
    #                  0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4,
    #                  0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5,
    #                  0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6,
    #                  0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7,
    #                  0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8,
    #                  0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9,
    #                  0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    
    


    # Determine number of processes to use
    if max_processes is None:
        max_processes = mp.cpu_count()-1  # Leave one core free
    
    # Limit processes to available options or CPU cores, whichever is smaller
    num_processes = min(max_processes, len(all_options))
    
    print(f"Running simulations on {num_processes} processes for {len(all_options)} options")
    print(f"Available CPU cores: {mp.cpu_count()}")
    
    # Prepare arguments for each process
    process_args = [(options, options_id, shorthand, H, Ns) for options_id, options in enumerate(all_options)]
    
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
    run_parallel_simulations(max_processes=5)
    