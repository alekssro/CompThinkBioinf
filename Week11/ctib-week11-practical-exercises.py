
# coding: utf-8

# In[1]:


import math  # Just ignore this :-)

def log(x):
    if x == 0:
        return float('-inf')
    return math.log(x)

#Functions needed for Viterbi decoding
def compute_w(model, x):
    """Computes the omega table
    Input: model=class 'hmm' object. x=indices of observations
    Output: w table of class 'list'"""
    k = len(model.init_probs)
    n = len(x)

    w = make_table(k, n)

    # Base case: fill out w[i][0] for i = 0..k-1
    for i in range(k):
        w[i][0] = model.init_probs[i] * model.emission_probs[i][x[0]]

    # Inductive case: fill out w[i][j] for i = 0..k, j = 0..n-1
    for j in range(1, n):
        for i in range(k):
            for t in range(k):
                w[i][j] = max(w[i][j], model.emission_probs[i][x[j]] * w[t][j-1] * model.trans_probs[t][i])

    return(w)

def backtrack(w, model, x):
    N = len(w[0])
    K = len(w)
    z = [0] * N
    max_ind = None
    max_path = 0

    #start with the state with higher probability in last column
    for i in range(K-1):
        if(max_path < w[i][N-1]):
            max_path = max(max_path, w[i][N-1])
            z[N-1] = i

    #check which state did we come from
    for n in range(N-2, -1, -1):
        for k in range(K):
            if(w[k][n] * model.emission_probs[z[n+1]][x[n+1]] * model.trans_probs[k][z[n+1]]) == w[z[n+1]][n+1]:
                z[n] = k
                break

    return z


# # CTiB - Week 11 - Practical Exercises

# In the exercise below, you will implement and experiment with the computation of the posterior decoding as explained in the lectures in week 10.

# # 1 - Background

# Below you will implement and experiment with posterior decoding. The implementation has been split into three parts:
#
# 1. Implement the forward algorithm, i.e. fill out the $\alpha$ table using the recursion presented at the lecture.
# 2. Implement the backward algorithm, i.e. fill out the $\beta$ table using the recursion presented at the lecture.
# 3. Using the $\alpha$ and $\beta$ tables to compute the posterior decoding as explained in class
#
# We'll be working with the 7-state model (`hmm_7_state`) model that we also worked with last time. The model is included below.

# In[2]:


class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs


# In[3]:


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


# We also need the helper functions for translating between observations/paths and indices.

# In[4]:


def translate_path_to_indices(path):
    return list(map(lambda x: int(x), path))

def translate_indices_to_path(indices):
    return ''.join([str(i) for i in indices])

def translate_observations_to_indices(obs):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    return [mapping[symbol.lower()] for symbol in obs]

def translate_indices_to_observations(indices):
    mapping = ['a', 'c', 'g', 't']
    return ''.join(mapping[idx] for idx in indices)


# Additionally, you're given the function below that constructs a table of a specific size filled with zeros.

# In[5]:


def make_table(m, n):
    """Make a table with `m` rows and `n` columns filled with zeros."""
    return [[0] * n for _ in range(m)]


# You'll be testing your code with the same two sequences as last time, i.e:

# In[6]:


x_short = 'GTTTCCCAGTGTATATCGAGGGATACTACGTGCATAGTAACATCGGCCAA'
z_short = '33333333333321021021021021021021021021021021021021'


# In[7]:


x_long = 'TGAGTATCACTTAGGTCTATGTCTAGTCGTCTTTCGTAATGTTTGGTCTTGTCACCAGTTATCCTATGGCGCTCCGAGTCTGGTTCTCGAAATAAGCATCCCCGCCCAAGTCATGCACCCGTTTGTGTTCTTCGCCGACTTGAGCGACTTAATGAGGATGCCACTCGTCACCATCTTGAACATGCCACCAACGAGGTTGCCGCCGTCCATTATAACTACAACCTAGACAATTTTCGCTTTAGGTCCATTCACTAGGCCGAAATCCGCTGGAGTAAGCACAAAGCTCGTATAGGCAAAACCGACTCCATGAGTCTGCCTCCCGACCATTCCCATCAAAATACGCTATCAATACTAAAAAAATGACGGTTCAGCCTCACCCGGATGCTCGAGACAGCACACGGACATGATAGCGAACGTGACCAGTGTAGTGGCCCAGGGGAACCGCCGCGCCATTTTGTTCATGGCCCCGCTGCCGAATATTTCGATCCCAGCTAGAGTAATGACCTGTAGCTTAAACCCACTTTTGGCCCAAACTAGAGCAACAATCGGAATGGCTGAAGTGAATGCCGGCATGCCCTCAGCTCTAAGCGCCTCGATCGCAGTAATGACCGTCTTAACATTAGCTCTCAACGCTATGCAGTGGCTTTGGTGTCGCTTACTACCAGTTCCGAACGTCTCGGGGGTCTTGATGCAGCGCACCACGATGCCAAGCCACGCTGAATCGGGCAGCCAGCAGGATCGTTACAGTCGAGCCCACGGCAATGCGAGCCGTCACGTTGCCGAATATGCACTGCGGGACTACGGACGCAGGGCCGCCAACCATCTGGTTGACGATAGCCAAACACGGTCCAGAGGTGCCCCATCTCGGTTATTTGGATCGTAATTTTTGTGAAGAACACTGCAAACGCAAGTGGCTTTCCAGACTTTACGACTATGTGCCATCATTTAAGGCTACGACCCGGCTTTTAAGACCCCCACCACTAAATAGAGGTACATCTGA'
z_long = '3333321021021021021021021021021021021021021021021021021021021021021021033333333334564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564563210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210321021021021021021021021033334564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564563333333456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456332102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102103210210210210210210210210210210210210210210210210210210210210210'


# Remember to translate these sequences to indices before using them with your algorithms.

# # 2 - The forward algorithm

# First, we will implement the forward algorithm.

# ## Computation of the $\alpha$ table

# Implement a function `compute_alpha` that fills out the alpha table cf. the recursion presented in the lecture.

# In[ ]:


def compute_alpha(model, x):
    K = len(model.init_probs)
    N = len(x)

    alpha = make_table(K, N)

    # Base case: fill out alpha[i][0] for i = 0..k-1
    for k in range(K):
        alpha[k][0] = model.init_probs[k] * model.emission_probs[k][x[1]]

    # Inductive case: fill out alpha[i][j] for i = 0..k, j = 1..n-1
    for n in range(1, N):
        for k in range(K):
            for t in range(K):
                alpha[k][n] += alpha[t][n-1] * model.trans_probs[t][k] * model.emission_probs[k][x[n]]

    return alpha

compute_alpha(hmm_7_state, translate_observations_to_indices(x_short))

# ## Using the $\alpha$ table to compute $p({\bf X})$

# Recall from the lecture that $p({\bf X}) = \sum_{{\bf z}_n} \alpha({\bf z}_n)$  , i.e. the sum of the entries in the rightmost coloum of the $\alpha$-table. Make a function `compute_px` that computes $p({\bf X})$ from the $\alpha$ table.

# Now, write a function that given the $\omega$-table, returns the probability of an optimal path through the HMM. As explained in the lecture, this corresponds to finding the highest probability in the last column of the table.

# In[ ]:


def compute_px(alpha):

    K = len(alpha)
    N = len(alpha[K-1])
    pX = 0

    # compute p(X) from the alpha table and return the value
    for k in range(K):
        pX += alpha[k][N-1]

    return(pX)

compute_px(compute_alpha(hmm_7_state, translate_observations_to_indices(x_short)))

# Now test your implementation by computing the probabillity of `x_short` and `x_long` under the 7-state model, i.e:

# In[ ]:


alpha_short = compute_alpha(hmm_7_state, translate_observations_to_indices(x_short))
print (compute_px(alpha_short))

alpha_long = compute_alpha(hmm_7_state, translate_observations_to_indices(x_long))
print (compute_px(alpha_long))


# What are the probabilities?

# # 3 - The backward algorithm

# In[ ]:


# Secondly, we will implement the backward algorithm.


# ## Computation of the $\beta$ table

# Implement a function `compute_beta` that fills out the alpha table cf. the recursion presented in the lecture.

# In[ ]:


def compute_beta(model, x):
    K = len(model.init_probs)
    N = len(x)

    beta = make_table(K, N)

    # Base case: fill out beta[k][N-1] for k = 0..K-1
    for k in range(K):
        beta[k][N-1] = 1

    # Inductive case: fill out beta[k][n] for k = K-1..0, n = N-2..0 (backwards)
    for n in range(N-2, -1, -1):
        for k in range(K-1, -1, -1):
            for t in range(K):
                beta[k][n] += beta[t][n+1] * model.trans_probs[k][t] * model.emission_probs[t][x[n+1]]


    return beta

compute_beta(hmm_7_state, translate_observations_to_indices(x_short))

# Test your implementation of the backward algorithm.

# In[ ]:


# Your code for testing compute_beta here


# # 4 - Posterior decoding

# Finally, combine your implementations for the forward- and backward algorithms to compute a posterior decoding. Make a function `posterior_decoding` that takes a model and a sequence of observations as input, and returns a posterior decoding of the sequence of inputs.

# In[ ]:


def posterior_decoding(model, x):

    K = len(model.init_probs)
    N = len(x)
    z = [None] * N

    alpha = compute_alpha(model, x)
    beta = compute_beta(model, x)
    pX = compute_px(alpha)

    for n in range(N):
        maxim = 0
        for k in range(K):
            if (maxim < (alpha[k][n] * beta[k][n] / pX)):
                z[n] = k
                maxim = alpha[k][n] * beta[k][n] / pX

    return z

print(posterior_decoding(hmm_7_state, translate_observations_to_indices(x_short)))

# Use the function `posterior_decoding` to compute a posterior decoding of `x_short` and `x_long` under the 7-state model. How does these decoding compare to the Viterbi decodings of these sequences under the 7-state model?

# In[ ]:


#Posterior decoding
post_short = posterior_decoding(hmm_7_state, translate_observations_to_indices(x_short))
#post_long = posterior_decoding(hmm_7_state, translate_observations_to_indices(x_long))
#calculating posterior decoding of x_long we get an error: ZeroDivisionError
#This is because the calculated numbers become too small to be represented and pX -> 0
#we would have to scale the values obtain in alpha and beta to solve this problem.

#Viterbi decoding
w = compute_w(hmm_7_state, translate_observations_to_indices(x_short))
z_viterbi = backtrack(w, hmm_7_state, translate_observations_to_indices(x_short))

post_short == z_viterbi
