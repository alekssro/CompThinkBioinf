
# coding: utf-8
# This is a compilation of used functions used to compute hidden Markov models

# log function to compute 0
import math  # Just ignore this :-)

def log(x):
    if x == 0:
        return float('-inf')
    return math.log(x)

def read_fasta_file(filename):
    """
    Reads the given FASTA file f and returns a dictionary of sequences.

    Lines starting with ';' in the FASTA file are ignored.
    """
    sequences_lines = {}
    current_sequence_lines = None
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(';') or not line:
                continue
            if line.startswith('>'):
                sequence_name = line.lstrip('>')
                current_sequence_lines = []
                sequences_lines[sequence_name] = current_sequence_lines
            else:
                if current_sequence_lines is not None:
                    current_sequence_lines.append(line)
    sequences = {}
    for name, lines in sequences_lines.items():
        sequences[name] = ''.join(lines)
    return sequences

# class hmm: hidden Markov model with an init, trans and emission probs matrix
class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs

########## Set of functions for translations #####################

def translate_indices_to_path_7state(indices):
    mapping = ['C', 'C', 'C', 'N', 'R', 'R', 'R']
    return ''.join([mapping[i] for i in indices])

def translate_indices_to_path_3state(indices):
    mapping = ['C', 'N', 'R']
    return ''.join([mapping[i] for i in indices])

def translate_observations_to_indices(obs):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    return [mapping[symbol.lower()] for symbol in obs]

def translate_indices_to_observations(indices):
    mapping = ['a', 'c', 'g', 't']
    return ''.join(mapping[idx] for idx in indices)

def translate_path_to_indices_3state(obs):
    mapping = {"C":0, "c":0,"N":1, "n":1,"R": 2,"r":2}
    return [mapping[symbol.lower()] for symbol in obs]

def translate_path_to_indices_7state(obs):
    mapping = {"ccc":"012", "n":"3", "rrr":"456"}
    return [s for s in [mapping[symbol.lower()] for symbol in obs]]

def translate_path_to_indices_7state_forloop(obs):
    lst1 = []
    c = 0
    r = 4

    for j in obs:
        if j is "N":
            lst1.append(3)
        if j is "C":
            lst1.append(c)
            if c != 2:
                c+=1
            else:
                c=0
        if j is "R":
            lst1.append(r)
            if r != 6:
                r+=1
            else:
                r=4

    return lst1

def make_table(m, n):
    """Make a table with `m` rows and `n` columns filled with zeros."""
    return [[0] * n for _ in range(m)]

#We need a table filled with -inf for log implementation
def make_table_log(m, n):
    """Make a table with `m` rows and `n` columns filled with -inf."""
    return [[float("-inf")] * n for _ in range(m)]

def viterbi_log(model, x):
    """Function that calculates the optimal path for a sequence of observations and a model
        Input: model = hmm class model; x = indices of sequence of observations
        Output: z = optimal path of states"""

    K = len(model.init_probs)
    N = len(x)

    ############# Calculate w matrix #############
    w = make_table_log(K, N)

    # Base case: fill out w[i][0] for i = 0..k-1
    for i in range(K):
        w[i][0] = log(model.init_probs[i]) + log(model.emission_probs[i][x[0]])

    # Inductive case: fill out w[i][j] for i = 0..k, j = 0..n-1
    for j in range(1, N):
        for i in range(K):
            for t in range(K):
                w[i][j] = max(w[i][j], log(model.emission_probs[i][x[j]]) + w[t][j-1] + log(model.trans_probs[t][i]))


    ############# Backtracking #############
    z = [None] * N
    max_ind = None
    max_path = float("-inf")

    #start with the state with higher probability in last column
    for i in range(K-1):
        if(max_path < w[i][N-1]):
            max_path = max(max_path, w[i][N-1])
            z[N-1] = i

    #check which state did we come from
    for n in range(N-2, -1, -1):
        for k in range(K):
            if(w[k][n] + log(model.emission_probs[z[n+1]][x[n+1]]) +
               log(model.trans_probs[k][z[n+1]])) == w[z[n+1]][n+1]:
                z[n] = k
                break

    return z

# Your implementations of compute_w_log and opt_path_prob_log from week 10
def compute_w_log(model, x):
    k = len(model.init_probs)
    n = len(x)

    w = make_table_log(k, n)

    # Base case: fill out w[i][0] for i = 0..k-1
    for i in range(k):
        w[i][0] = log(model.init_probs[i]) + log(model.emission_probs[i][x[0]])

    # Inductive case: fill out w[i][j] for i = 0..k, j = 0..n-1
    for j in range(1, n):
        for i in range(k):
            for t in range(k):
                w[i][j] = max(w[i][j], log(model.emission_probs[i][x[j]]) + w[t][j-1] + log(model.trans_probs[t][i]))

    return(w)

def opt_path_prob_log(w):
    k = len(w)
    n = len(w[k-1])
    max_path = float("-inf")

    for i in range(k-1):
        max_path = max(max_path, w[i][n-1])

    return max_path

def backtrack_log(w, model, x):
    N = len(w[0])
    K = len(w)
    z = [None] * N
    max_ind = None
    max_path = float("-inf")

    #start with the state with higher probability in last column
    for i in range(K-1):
        if(max_path < w[i][N-1]):
            max_path = max(max_path, w[i][N-1])
            z[N-1] = i

    #check which state did we come from
    for n in range(N-2, -1, -1):
        for k in range(K):
            if(w[k][n] + log(model.emission_probs[z[n+1]][x[n+1]]) +
               log(model.trans_probs[k][z[n+1]])) == w[z[n+1]][n+1]:
                z[n] = k
                break

    return z

# Compute accuracy of a prediction
def compute_accuracy(true_ann, pred_ann):
    if len(true_ann) != len(pred_ann):
        return 0.0
    return sum(1 if true_ann[i] == pred_ann[i] else 0
               for i in range(len(true_ann))) / len(true_ann)

# Function that trains a model of k states, D observables given an x an z
def training_by_counting(K, D, x, z):

    matrix_trans = make_table(K,K)
    matrix_emission = make_table(K,D)
    N = len(x)

    matrix_init = [0 for i in range(K)]
    matrix_init[z[0]] = 1

    #transition probs matrix calculation
    for i in range(len(z)-1):
        curr_state = z[i]
        next_state = z[i+1]
        matrix_trans[curr_state][next_state] += 1

    #Make list of sums of rows in matrix
    lst_sum = []
    for lst in matrix_trans:
        lst_sum.append(sum(lst))

    #Divide all values in list in matrix with the corresponding
    #index in the list of sums.
    for i in range(K):
        for j in range(K):
            matrix_trans[i][j] = matrix_trans[i][j] / lst_sum[i]

    #emission probs matrix calculation
    for n in range(N):
        matrix_emission[z[n]][x[n]] +=1

        #Make list of sums of rows in matrix
    lst_sum = []
    for lst in matrix_emission:
        lst_sum.append(sum(lst))

    #Divide all values in list in matrix with the corresponding
    #index in the list of sums.
    for i in range(K):
        for j in range(D):
            matrix_emission[i][j] = matrix_emission[i][j] / lst_sum[i]

    return hmm(matrix_init,matrix_trans,matrix_emission)
