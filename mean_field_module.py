import numpy as np
from numba import float64, int64, njit
from numba.experimental import jitclass
# from numba.typed import List

#%% UNUSED BASICS
# def make_accumulator(n):
#     acc = List()
#     for _ in range(n):
#         acc.append(List.empty_list(float))  # or int, etc.
#     return acc
#%% BASICS

@njit
def logsumexp_1d_masked(x):
    maxv = -np.inf
    for i in range(x.shape[0]):
        xi = x[i]
        if xi > maxv:
            maxv = xi

    if maxv == -np.inf:
        return -np.inf

    s = 0.0
    for i in range(x.shape[0]):
        xi = x[i]
        if xi > -np.inf:
            s += np.exp(xi - maxv)

    return maxv + np.log(s)

@njit
def logaddexp(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    if a > b:
        return a + np.log1p(np.exp(b - a))
    else:
        return b + np.log1p(np.exp(a - b))

# supposedly suboptimal compute_log_action_probs function replaced with better one below
# @njit
# def compute_log_action_probs(population, q_s):
#     n = len(population)
#     action_probs = np.full(3, -np.inf)
    
#     # accumulate log-probabilities
#     for i in range(n):
#         log_prop = population[i]
#         if log_prop == -np.inf:
#             continue
#         for a in range(3):
#             action_probs[a] = logaddexp(action_probs[a], log_prop + q_s[i,a])
    
#     # optional normalization (if population is not normalized, or for extra safety)
#     # if normalize:
#     maxv = -np.inf
#     for a in range(3):
#         if action_probs[a] > maxv:
#             maxv = action_probs[a]

#     total = 0.0
#     for a in range(3):
#         if action_probs[a] > -np.inf:
#             total += np.exp(action_probs[a] - maxv)

#     logZ = maxv + np.log(total)

#     for a in range(3):
#         action_probs[a] -= logZ
#     return action_probs

@njit
def compute_log_action_probs(population, q_s):
    n = population.shape[0]

    # accumulate in linear space
    acc0 = 0.0
    acc1 = 0.0

    for i in range(n):
        log_pi = population[i]
        if log_pi == -np.inf:
            continue

        # per-state max for stability
        m = q_s[i, 0]
        if q_s[i, 1] > m:
            m = q_s[i, 1]

        w = np.exp(log_pi + m)

        acc0 += w * np.exp(q_s[i, 0] - m)
        acc1 += w * np.exp(q_s[i, 1] - m)

    # normalize
    total = acc0 + acc1

    out = np.empty(2)
    out[0] = np.log(acc0 / total)
    out[1] = np.log(acc1 / total)

    return out

@njit
def get_random_log_population(q_s):
    
    population = np.full(len(q_s), -np.inf)
    n  = len(q_s)
    # Generate random values and normalize in log space
    v = np.random.normal(0, 1, n)
    # Take absolute values to ensure non-negative probabilities
    v = np.abs(v)
    # Convert to linear space, normalize, then back to log space
    v_norm = v / np.sum(v)
    for i in range(n):
        population[i] = np.log(v_norm[i]) if v_norm[i] > 0 else -np.inf
    return population

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

def reverse_shift_vectorized(Finv):
    """
    Vectorized version to compute (choice, nn_choice) for many transitions.

    Args:
        Finv: dict mapping integers to state strings

    Returns:
        choices: array of shape (n), integers
        nn_choices: array of shape (n), integers
    """
    n = len(Finv)
    target_states = np.arange(n)  

    # Convert integer keys to strings
    target_strs = np.array([Finv[t] for t in target_states])

    # Prepare arrays for results
    choices = np.zeros(n, dtype=int)
    nn_choices = np.zeros(n, dtype=int)

    # Vectorized extraction: last two characters of target strings
    for i in range(n):
        s = target_strs[i]
        if len(s) < 2:
            print("Empty memory state encountered in reverse_shift_vectorized.")
            # default to valid binary choice to avoid out-of-bounds
            choices[i] = 0
            nn_choices[i] = 0
            continue
        # choice = second last char, nn_choice = last char
        choices[i] = int(s[-2])
        nn_choices[i] = int(s[-1])

    return choices, nn_choices


def state_transitions(q_H, H, F, Finv):

    n = len(q_H)

    P_key   = np.zeros((n, n, 4), dtype=int)
    P_value = np.zeros((n, n, 4), dtype=float)

    for i in range(n):
        for j in range(n):

            key_i = Finv[i]
            key_j = Finv[j]

            # 2-way probability vectors
            pi = q_H[key_i]   # e.g. [p0, p1]
            pj = q_H[key_j]

            idx = 0
            for a in range(2):          # output of i
                for b in range(2):      # output of j

                    # next state for i
                    next_i = F[shift(key_i, a, b, H)]

                    P_key[i,j,idx] = next_i
                    P_value[i,j,idx] = pi[a] + pj[b]

                    idx += 1

    return P_key, P_value

def CM_state_transitions(q_H, H, F, Finv, committment_signal):

    n = len(q_H)

    P_key   = np.zeros((n, 4), dtype=int)
    P_value = np.zeros((n, 4), dtype=float)

    for i in range(n):
        key_i = Finv[i]

        # 2-way probability vectors
        pi = q_H[key_i]   # e.g. [p0, p1]

        for a in range(2):          # output of i
            # agent j is committed to committment_signal
            b = committment_signal
            idx = a * 2 + b
            # next state for i
            next_i = F[shift(key_i, a, b, H)]

            P_key[i,idx] = next_i
            P_value[i,idx] = pi[a] 

    return P_key, P_value

def integer_probabilities(q_H, F):
    n = len(q_H)
    q_s = np.zeros((n, 2))
    for k in F:
        i = F[k]
        # print(i, k, q_H[k])
        q_s[i,:] = q_H[k]        # now a vector [p0, p1]
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
    ('q_s', float64[:, :]),
    ('population', float64[:]),
    ('P_key', int64[:,:,:]),
    ('P_value', float64[:,:,:]),
    # ('CM_P_key', int64[:, :]),
    # ('CM_P_value', float64[:, :]),
    ('empty_state', int64),
    ('steady_state', int64),
    ('CM', float64),
    ('committment_signal', int64),
    ('population_log_probs', float64[:]),
    ('choices', int64[:]),
    ('nn_choices', int64[:]),
    ('dt', float64)
]

@jitclass(spec)
class LLM_dynamics_integer:
    """
        Simple class simulator
    """
    def __init__(self, q_s, P_key, P_value, choices, nn_choices, empty_state, steady_state, CM, committment_signal, dt, population = None):
        self.q_s = q_s
        self.P_key = P_key
        self.P_value = P_value
        self.choices = choices
        self.nn_choices = nn_choices
        self.empty_state = empty_state
        self.CM = CM
        self.steady_state = steady_state
        self.committment_signal = committment_signal
        self.dt = dt
        # self.CM_P_key = CM_P_key
        # self.CM_P_value = CM_P_value
        if population is None:
            self.population = get_random_log_population(self.q_s)
            # print("--- WARNING: population vector not set. Initializing random population distribution... ---")
        else:
            self.population = population
        self.population_log_probs = compute_log_action_probs(self.population, self.q_s)
        
    def initialize_population_zero(self):
        """
            Initialize the population of LLMs to the empty state
        """
        
        for i in range(len(self.population)):
            self.population[i] = -np.inf
            if i == self.empty_state:
                self.population[i] = 0.0
        self.population_log_probs = compute_log_action_probs(self.population, self.q_s)
        # print("Initialized population to empty state.")

    def initialize_population_random(self):
        """
            Initialize the population of LLMs
        """

        self.population = get_random_log_population(self.q_s)
        self.population_log_probs = compute_log_action_probs(self.population, self.q_s)
        # print("Initialized population to random distribution.")
    
    def initialize_population_steady(self):
        """
            Initialize the population of LLMs to a steady state
        """
        for i in range(len(self.population)):
            self.population[i] = -np.inf
            if i == self.steady_state:
                self.population[i] = 0.0
        self.population_log_probs = compute_log_action_probs(self.population, self.q_s)
        # print("Initialized population to steady state.")

    def get_update_stats(self):
        
        action_log_probs = compute_log_action_probs(self.population, self.q_s)
        return action_log_probs
        
    # def math_update(self):
    #     n = self.population.shape[0]
    #     new_population_lin = np.zeros(n)

    #     has_committed = self.CM > 0.0
    #     if has_committed:
    #         log_CM = np.log(self.CM)
    #         log_nonCM = np.log(1.0 - self.CM)

    #     # precompute non-committed weights
    #     w = np.zeros(n)
    #     for j in range(n):
    #         pj = self.population[j]
    #         if pj > -np.inf:
    #             w[j] = np.exp(pj + (log_nonCM if has_committed else 0.0))

    #     for i in range(n):
    #         pi = self.population[i]
    #         if pi == -np.inf:
    #             continue

    #         pi_lin = np.exp(pi + (log_nonCM if has_committed else 0.0))

    #         for k in range(9):

    #             # committed interaction
    #             if has_committed and (k % 3) == self.committment_signal:
    #                 next_i = self.CM_P_key[i, k]
    #                 new_population_lin[next_i] += np.exp(log_CM + self.CM_P_value[i, k])

    #             # folded non-committed interactions
    #             for j in range(n):
    #                 if w[j] == 0.0:
    #                     continue

    #                 next_i = self.P_key[i, j, k]
    #                 new_population_lin[next_i] += pi_lin * w[j] * np.exp(self.P_value[i, j, k])

    #     # convert back to log space + normalize
    #     new_population = np.full(n, -np.inf)
    #     total = 0.0
    #     for s in range(n):
    #         total += new_population_lin[s]

    #     logZ = np.log(total)

    #     for s in range(n):
    #         if new_population_lin[s] > 0.0:
    #             new_population[s] = np.log(new_population_lin[s]) - logZ

    #     return new_population


    def whole_population_log_action_probs(self):
        # Compute action probabilities from uncommitted_population
        #uncommitted_action_log_probs = compute_log_action_probs(self.population, self.q_s)
        uncommitted_action_log_probs = self.population_log_probs.copy()
        # check committed population
        has_committed = self.CM > 0.0
        if has_committed:
            log_CM = np.log(self.CM)
            non_committed_proportion = np.log(1 - self.CM)
            # initialise final action logprobs vector
            final_action_log_probs = np.full(2, -np.inf)
            # add committed contribution
            for a in range(2):
                committed_log_prob = -np.inf if a != self.committment_signal else 0.0
                log_committed_contribution = log_CM + committed_log_prob
                log_non_committed_contribution = non_committed_proportion + uncommitted_action_log_probs[a]
                final_action_log_probs[a] = logaddexp(log_committed_contribution, log_non_committed_contribution)
            return final_action_log_probs
        else:
            return uncommitted_action_log_probs
    
    def algorithmic_update(self):
        # initialize new population
        new_population = np.full(self.population.shape[0], -np.inf)
        # get whole population action log probs (both committed and non-committed)
        whole_population_log_action_probs = self.whole_population_log_action_probs()
        for source_state, source_state_log_prob in enumerate(self.population):
            if source_state_log_prob == -np.inf:
                continue
            
            # get all possible transition targets from this source state
            transition_targets = self.P_key[source_state, 0, :]  # shape (4,)

            # iterate over all possible target states. There should be 9 targets per source state.
            for target_state in transition_targets:
                # get choice and nn_choice for this transition
                choice, nn_choice = self.choices[target_state], self.nn_choices[target_state]
                # get probability of transitioning from source_state to target_state based on population and transition constraints
                transition_log_prob = self.q_s[source_state, choice]  + whole_population_log_action_probs[nn_choice]
                
                # update probability of target state, using probability of observing source state and then transitioning from source state to target state
                new_population[target_state] = logaddexp(new_population[target_state], source_state_log_prob + transition_log_prob)
            
        # normalize new population using trick from compute_log_action_probs
        maxv = -np.inf
        for i in range(new_population.shape[0]):
            if new_population[i] > maxv:
                maxv = new_population[i]
        total = 0.0
        for i in range(new_population.shape[0]):
            if new_population[i] > -np.inf:
                total += np.exp(new_population[i] - maxv)
        logZ = maxv + np.log(total)
        for i in range(new_population.shape[0]):
            if new_population[i] > -np.inf:
                new_population[i] -= logZ
        
        # update population
        self.population = new_population

        # update population log probs
        self.population_log_probs = compute_log_action_probs(self.population, self.q_s)
            
    def algorithmic_update_asynchronous_temp(self):
            """
            Asynchronously updates the population state over continuous time increment dt,
            operating entirely in log-space to preserve numerical precision.
            """
            n_states = self.population.shape[0]
            
            # Precompute log(dt) and log(1 - dt) for the Euler step weighting
            # Handle the dt=1.0 case safely to avoid log(0) warnings
            log_dt = np.log(self.dt) if self.dt > 0.0 else -np.inf
            log_1_minus_dt = np.log(1.0 - self.dt) if self.dt < 1.0 else -np.inf
            
            # 1. Get whole population action probabilities (returns in log space)
            whole_population_log_action_probs = self.whole_population_log_action_probs()
            
            # 2. Compute incoming gain rates for each target state directly in log space
            # log_gains = np.full(n_states, -np.inf)
            new_population = np.full(n_states, -np.inf)

            for source_state in range(n_states):
                source_state_log_prob = self.population[source_state]
                if source_state_log_prob == -np.inf:
                    continue
                    
                # Get target states (matching exact indexing from original script)
                transition_targets = self.P_key[source_state, 0, :]
                
                for target_state in transition_targets:
                    choice = self.choices[target_state]
                    nn_choice = self.nn_choices[target_state]
                    
                    # Transition path probability: log( P(source) * P(transition) )
                    transition_log_prob = self.q_s[source_state, choice] + whole_population_log_action_probs[nn_choice]
                    path_log_prob = source_state_log_prob + transition_log_prob
                    
                    # Accumulate gains in log space using the numerically stable logaddexp
                    log_gains = logaddexp(new_population[target_state], path_log_prob)
                    retained_log_prob = log_1_minus_dt + self.population[target_state]
                    new_population[target_state] = logaddexp(retained_log_prob, log_dt + log_gains)
                    # log_gains[target_state] = logaddexp(log_gains[target_state], path_log_prob)
                    
            # 3. Asynchronous ODE Euler Update in log space
            # Mathematically: x_new = (1 - dt)*x + dt*gains
            # new_population = np.full(n_states, -np.inf)
            # for k in range(n_states):
            #     retained_log_prob = log_1_minus_dt + self.population[k]
            #     gained_log_prob = log_dt + log_gains[k]
                
            #     new_population[k] = logaddexp(retained_log_prob, gained_log_prob)
                
            # 4. Normalize the new population using the logsumexp trick 
            # (borrowed from the original algorithmic_update to ensure exact consistency)
            maxv = -np.inf
            for i in range(n_states):
                if new_population[i] > maxv:
                    maxv = new_population[i]
                    
            if maxv > -np.inf:
                total = 0.0
                for i in range(n_states):
                    if new_population[i] > -np.inf:
                        total += np.exp(new_population[i] - maxv)
                logZ = maxv + np.log(total)
                
                for i in range(n_states):
                    if new_population[i] > -np.inf:
                        self.population[i] = new_population[i] - logZ
                    else:
                        self.population[i] = -np.inf
            else:
                # Fallback if population completely zeros out
                for i in range(n_states):
                    self.population[i] = -np.inf

            # 5. Update action probability cache
            self.population_log_probs = compute_log_action_probs(self.population, self.q_s)

    def algorithmic_update_asynchronous(self):
        number_of_steps = int(1.0 / self.dt)
        for _ in range(number_of_steps):
            self.algorithmic_update_asynchronous_temp()
        
    # def math_update(self):
    #     """
    #     Update the population of LLMs one Monte Carlo time step.
    #     population : (n,) log proportions of all states
    #     P_key      : (n, n, 9) next-state indices
    #     P_value    : (n, n, 9) log probabilities
    #     CM_P_key     : (n, 9) next-state indices for committed interactions
    #     CM_P_value    : (n, 9) log probabilities for committed interactions

    #     class parameters:
    #     CM         : fraction of population committed
    #     commitment_signal : index of the committed output

    #     Returns: updated log-populations of **non-committed population**, normalized
    #     """

    #     # check if committed population exists
    #     has_committed = self.CM > 0.0
    #     n = self.population.shape[0]
    #     if has_committed:
    #         log_CM = np.log(self.CM)
    #         log_nonCM = np.log(1 - self.CM)

    #     # initialize new population vector
    #     new_population = np.full(n, -np.inf)

    #     # create an array to track contributions for each population state. Each array element will be another array of contributions from different interactions.

    #     for i in range(n):
    #         # get log_prob of observing uncommitted source state i
    #         pi = self.population[i] 
    #         if pi == -np.inf:   
    #             continue
    #         if has_committed: pi += log_nonCM  # scale by non-committed fraction

    #         # each source state has k=9 possible transitions (which may include staying in the same state)
    #         for k in range(9):
                
    #             # add contibution from committed interactions
    #             if has_committed and (k % 3) == self.committment_signal:
    #                 # get next i state from committed interaction with committment_signal
    #                 next_i = self.CM_P_key[i, k]
    #                 # total contribution to next_i from i interacting with committed agent
    #                 ## contribution corresponds to log-probability of observing an interaction between i and a committed agent,
    #                 ## and the interaction producing the outputs that lead to this transition
    #                 contrib = log_CM + self.CM_P_value[i, k]

    #                 # add to new population
    #                 new_population[next_i] = logaddexp(new_population[next_i], contrib)

    #             # add contribution from non-committed interactions
    #             for j in range(n):
    #                 # get log_prob of interacting with state j
    #                 pj = self.population[j]
    #                 if pj == -np.inf:
    #                     continue
    #                 if has_committed: pj += log_nonCM

    #                 # get kth possible state transition from i after i interacts with j
    #                 next_i = self.P_key[i, j, k]

    #                 # total contribution to next_i from i interacting with j.
    #                 ## this is the log-probability of observing an interaction between i and j,
    #                 ## and the interaction producing the outputs that lead to this transition
    #                 contrib = pi + pj + self.P_value[i, j, k]

    #                 # add to new population
    #                 new_population[next_i] = logaddexp(new_population[next_i], contrib)

    #     # normalize only over non-committed population
    #     # Normalize the distribution in log space
    #     non_inf_values = np.array([v for v in new_population if v > -np.inf])
    #     if non_inf_values.size > 0:  # Check if any non-zero probabilities exist
    #         log_sum = logsumexp_1d_masked(non_inf_values)
    #         for state in range(n):
    #             if new_population[state] > -np.inf:
    #                 new_population[state] -= log_sum

    #     self.population = new_population