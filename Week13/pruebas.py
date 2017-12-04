
# coding: utf-8

import math  # Just ignore this :-)
import time

def log(x):
    if x == 0:
        return float('-inf')
    return math.log(x)


# # CTiB - Week 13 - Practical Exercises

# In the exercise below, you will implement and experiment with an example of how to apply a HMM for identifying coding regions (genes) in genetic matrial.

# # 1 - Using a HMM for Gene Finding
#
# Below we will investigate how to use a hidden Markov model for gene finding in prokaryotes.
#
# You are give a data set containing 2 Staphylococcus genomes, each containing several genes (i.e. substrings) obeying the "gene syntax" explained in class. The genomes are between 1.8 million and 2.8 million nucleotides.
#
# The genomes and their annontations are given in [FASTA format](https://en.wikipedia.org/wiki/FASTA_format).


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


# You can use the function like this (note that reading the entire genome will take some time):


g1 = read_fasta_file('genome1.fa')
print(g1['genome1'][:50])
g2 = read_fasta_file('genome2.fa')
print(g2['genome2'][:50])


# The data is:
#
# * The files [genome1.fa](https://users-cs.au.dk/cstorm/courses/ML_e17/projects/handin3/genome1.fa) and  [genome2.fa](https://users-cs.au.dk/cstorm/courses/ML_e17/projects/handin3/genome2.fa) contain the 2 genomes.
# * The files [true-ann1.fa](https://users-cs.au.dk/cstorm/courses/ML_e17/projects/handin3/true-ann1.fa) and [true-ann2.fa](https://users-cs.au.dk/cstorm/courses/ML_e17/projects/handin3/true-ann2.fa) contain the annotation of the two genomes with the tru gene structure. The annotation is given in FASTA format as a sequence over the symbols `N`, `C`, and `R`. The symbol `N`, `C`, or `R` at position $i$ in `true-annk.fa` gives the "state" of the nucleotide at position $i$ in `genomek.fa`. `N` means that the nucleotide is non-coding. `C` means that the nucleotide is coding and part of a gene in the direction from left to right. `R` means that the nucleotide is coding and part of gene in the reverse direction from right to left.
#
# The annotation files can also be read with `read_fasta_file`.
#
# You are given the same 7-state and 3-state HMM that you used before and similar helper functions:


class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs

init_probs_7_state = [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00]

trans_probs_7_state = [
    [0.00, 0.00, 0.90, 0.10, 0.00, 0.00, 0.00],
    [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.05, 0.90, 0.05, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
    [0.00, 0.00, 0.00, 0.10, 0.90, 0.00, 0.00],
]

emission_probs_7_state = [
    #   A     C     G     T
    [0.30, 0.25, 0.25, 0.20],
    [0.20, 0.35, 0.15, 0.30],
    [0.40, 0.15, 0.20, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.20, 0.40, 0.30, 0.10],
    [0.30, 0.20, 0.30, 0.20],
    [0.15, 0.30, 0.20, 0.35],
]

hmm_7_state = hmm(init_probs_7_state, trans_probs_7_state, emission_probs_7_state)


init_probs_3_state = [0.00, 0.10, 0.00]

trans_probs_3_state = [
    [0.90, 0.10, 0.00],
    [0.05, 0.90, 0.05],
    [0.00, 0.10, 0.90],
]

emission_probs_3_state = [
    #   A     C     G     T
    [0.40, 0.15, 0.20, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.20, 0.40, 0.30, 0.10],
]

hmm_3_state = hmm(init_probs_3_state, trans_probs_3_state, emission_probs_3_state)


# Notice that this time the function `translate_indices_to_path` that you have used in previous exervises is a bit different. In the 7-state model the states 0, 1, 2 represent coding (C), state 3 represents non-coding (N) and states 4, 5, 6 represent reverse-coding (R) as explained in class. This translation is done by the function `translate_indices_to_path_7state`. In the 3-state model the state 0 represents coding (C), state 1 represents non-coding (N) and state 2 represents reverse-coding (R) as explained in class. This translation is done by the function `translate_indices_to_path_3state`.

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
    mapping = {"c": 0, "n": 1, "r": 2}
    return [mapping[symbol.lower()] for symbol in obs]

def translate_path_to_indices_7state(obs):
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


# Now insert your Viterbi implementation (log transformed) in the cell below, this means that you should copy `compute_w_log`, `opt_path_prob_log`, `backtrack_log` and any other functions you may have defined yourself for your Viterbi implementation.

# Your implementations of compute_w_log and opt_path_prob_log from week 10
def viterbi_log(model, x):
    """Function that calculates the optimal path for a sequence of observations and a model
        Input: model = hmm class model; x = indices of sequence of observations
        Output: z = optimal path of states"""

    K = len(model.init_probs)
    N = len(x)

    ############# log probs in model #############
    emission_probs = make_table(K, len(model.emission_probs[0]))
    trans_probs = make_table(K, K)
    # init
    init_probs = [log(y) for y in model.init_probs]
    # emission
    for i in range(K):
        for j in range(len(model.emission_probs[i])):
            emission_probs[i][j] = log(model.emission_probs[i][j])

    #transition
    for i in range(K):
        for j in range(K):
            trans_probs[i][j] = log(model.trans_probs[i][j])

    ############# Calculate w matrix #############
    w = make_table_log(K, N)

    # Base case: fill out w[i][0] for i = 0..k-1
    for i in range(K):
        w[i][0] = init_probs[i] + emission_probs[i][x[0]]

    # Inductive case: fill out w[i][j] for i = 0..k, j = 0..n-1
    for j in range(1, N):
        for i in range(K):
            for t in range(K):
                w[i][j] = max(w[i][j], emission_probs[i][x[j]] + w[t][j-1] + trans_probs[t][i])


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
            if(w[k][n] + emission_probs[z[n+1]][x[n+1]] +
               trans_probs[k][z[n+1]]) == w[z[n+1]][n+1]:
                z[n] = k
                break

    return z


# Your code to get hmm_7_state_genome1 using training by counting on genome 1,
# predict an annotation of genome 2, and compare the prediction to true-ann2.fa
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

    return hmm(matrix_init, matrix_trans, matrix_emission)


true_ann1_7state = read_fasta_file('true-ann1.fa')
true_ann1_7state = true_ann1_7state['true-ann1']

# Get the training set
x = translate_observations_to_indices(g1["genome1"])
z = translate_path_to_indices_7state(true_ann1_7state)

# Get the model
hmm_7_state_genome1 = training_by_counting(7, 4, x, z)

import numpy as np
np.set_printoptions(precision=5)
print(hmm_7_state_genome1.init_probs)
print(np.matrix(hmm_7_state_genome1.trans_probs))
print(np.matrix(hmm_7_state_genome1.emission_probs))

true_ann1_3state = read_fasta_file('true-ann1.fa')
true_ann1_3state = true_ann1_3state['true-ann1']

# Get the training set
x = translate_observations_to_indices(g1["genome1"])
z = translate_path_to_indices_3state(true_ann1_3state)

# Get the model
hmm_3_state_genome1 = training_by_counting(3, 4, x, z)

import numpy as np
np.set_printoptions(precision=5)
print(hmm_3_state_genome1.init_probs)
print(np.matrix(hmm_3_state_genome1.trans_probs))
print(np.matrix(hmm_3_state_genome1.emission_probs))
