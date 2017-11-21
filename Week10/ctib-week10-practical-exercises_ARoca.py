
# coding: utf-8

# In[137]:


import math  # Just ignore this :-)

def log(x):
    if x == 0:
        return float('-inf')
    return math.log(x)


# # CTiB - Week 10 - Practical Exercises

# In the exercise below, you will implement and experiment with the computation of the Viterbi decoding as explained in the lectures in week 10.

# # 1 - Viterbi Decoding

# Below you will implement and experiment with the Viterbi algorithm. The implementation has been split into three parts:
# 
# 1. Fill out the $\omega$ table using the recursion presented at the lecture.
# 2. Find the state with the highest probability after observing the entire sequence of observations.
# 3. Backtrack from the state found in the previous step to obtain the optimal path.
# 
# We'll be working with the two models (`hmm_7_state` and `hmm_3_state`) that we also worked with last time: the 3 and 7-state models. We have included the models below.

# In[138]:


class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs


# In[139]:


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


# In[140]:


init_probs_3_state = [0.10, 0.80, 0.10]

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


# We also need the helper functions for translating between observations/paths and indices.

# In[141]:


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

# In[142]:


def make_table(m, n):
    """Make a table with `m` rows and `n` columns filled with zeros."""
    return [[0] * n for _ in range(m)]


# You'll be testing your code with the same two sequences as last time, i.e:

# In[143]:


x_short = 'GTTTCCCAGTGTATATCGAGGGATACTACGTGCATAGTAACATCGGCCAA'
z_short = '33333333333321021021021021021021021021021021021021'


# In[144]:


x_long = 'TGAGTATCACTTAGGTCTATGTCTAGTCGTCTTTCGTAATGTTTGGTCTTGTCACCAGTTATCCTATGGCGCTCCGAGTCTGGTTCTCGAAATAAGCATCCCCGCCCAAGTCATGCACCCGTTTGTGTTCTTCGCCGACTTGAGCGACTTAATGAGGATGCCACTCGTCACCATCTTGAACATGCCACCAACGAGGTTGCCGCCGTCCATTATAACTACAACCTAGACAATTTTCGCTTTAGGTCCATTCACTAGGCCGAAATCCGCTGGAGTAAGCACAAAGCTCGTATAGGCAAAACCGACTCCATGAGTCTGCCTCCCGACCATTCCCATCAAAATACGCTATCAATACTAAAAAAATGACGGTTCAGCCTCACCCGGATGCTCGAGACAGCACACGGACATGATAGCGAACGTGACCAGTGTAGTGGCCCAGGGGAACCGCCGCGCCATTTTGTTCATGGCCCCGCTGCCGAATATTTCGATCCCAGCTAGAGTAATGACCTGTAGCTTAAACCCACTTTTGGCCCAAACTAGAGCAACAATCGGAATGGCTGAAGTGAATGCCGGCATGCCCTCAGCTCTAAGCGCCTCGATCGCAGTAATGACCGTCTTAACATTAGCTCTCAACGCTATGCAGTGGCTTTGGTGTCGCTTACTACCAGTTCCGAACGTCTCGGGGGTCTTGATGCAGCGCACCACGATGCCAAGCCACGCTGAATCGGGCAGCCAGCAGGATCGTTACAGTCGAGCCCACGGCAATGCGAGCCGTCACGTTGCCGAATATGCACTGCGGGACTACGGACGCAGGGCCGCCAACCATCTGGTTGACGATAGCCAAACACGGTCCAGAGGTGCCCCATCTCGGTTATTTGGATCGTAATTTTTGTGAAGAACACTGCAAACGCAAGTGGCTTTCCAGACTTTACGACTATGTGCCATCATTTAAGGCTACGACCCGGCTTTTAAGACCCCCACCACTAAATAGAGGTACATCTGA'
z_long = '3333321021021021021021021021021021021021021021021021021021021021021021033333333334564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564563210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210321021021021021021021021033334564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564563333333456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456332102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102103210210210210210210210210210210210210210210210210210210210210210'


# Remember to translate these sequences to indices before using them with your algorithms.

# ## Implementing without log-transformation

# First, we will implement the algorithm without log-transformation. This will cause issues with numerical stability (like above when computing the joint probability), so we will use the log-transformation trick to fix this in the next section.

# ### Computation of the $\omega$ table

# In[145]:


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

compute_w(hmm_7_state, translate_observations_to_indices(x_short))


# ### Finding the joint probability of an optimal path

# Now, write a function that given the $\omega$-table, returns the probability of an optimal path through the HMM. As explained in the lecture, this corresponds to finding the highest probability in the last column of the table.

# In[146]:


def opt_path_prob(w):
    k = len(w)
    n = len(w[k-1])
    max_path = 0

    for i in range(k-1):
        max_path = max(max_path, w[i][n-1])
    
    return max_path


# Now test your implementation in the box below:

# In[147]:


w = compute_w(hmm_7_state, translate_observations_to_indices(x_short))
opt_path_prob(w)


# Now do the same for `x_long`. What happens?

# In[148]:


w = compute_w(hmm_7_state, translate_observations_to_indices(x_long))
opt_path_prob(w)

#We get a highest probability of 0 in the last column as the numbers are too small


# ### Obtaining an optimal path through backtracking

# Implement backtracking to find a most probable path of hidden states given the $\omega$-table.

# In[159]:


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


# In[160]:


w = compute_w(hmm_7_state, translate_observations_to_indices(x_short))
z_viterbi = backtrack(w, hmm_7_state, translate_observations_to_indices(x_short))

# w.index(opt_path_prob(w))
print(z_viterbi)
print(z_short)


# Now do the same for `x_long`. What happens?

# In[161]:


w = compute_w(hmm_7_state, translate_observations_to_indices(x_long))
z_viterbi = backtrack(w, hmm_7_state, translate_observations_to_indices(x_long))

# w.index(opt_path_prob(w))
print(z_viterbi)
print(z_long)

# It doesn't find where it came from because at some point probabilities low to 0


# ## Implementing with log-transformation

# Now implement the Viterbi algorithm with log transformation. The steps are the same as above.

# In[119]:


#We need a table filled with -inf
def make_table_log(m, n):
    """Make a table with `m` rows and `n` columns filled with -inf."""
    return [[float("-inf")] * n for _ in range(m)]


# ### Computation of the $\omega$ table

# In[120]:


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


# ### Finding the (log transformed) joint probability of an optimal path

# In[121]:


def opt_path_prob_log(w):
    k = len(w)
    n = len(w[k-1])
    max_path = float("-inf")

    for i in range(k-1):
        max_path = max(max_path, w[i][n-1])
    
    return max_path


# In[122]:


w = compute_w_log(hmm_7_state, translate_observations_to_indices(x_short))
opt_path_prob_log(w)


# Now do the same for `x_long`. What happens?

# In[123]:


w = compute_w_log(hmm_7_state, translate_observations_to_indices(x_long))
opt_path_prob_log(w)


# ### Obtaining an optimal path through backtracking

# In[124]:


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


# In[125]:


w = compute_w_log(hmm_7_state, translate_observations_to_indices(x_short))
z_viterbi_log = backtrack_log(w, hmm_7_state, translate_observations_to_indices(x_short))
print(z_viterbi_log)
print(z_viterbi)
z_viterbi == z_viterbi_log


# Now do the same for `x_long`. What happens?

# In[126]:


w = compute_w_log(hmm_7_state, translate_observations_to_indices(x_long))
z_viterbi_log = backtrack_log(w, hmm_7_state, translate_observations_to_indices(x_long))
print(z_viterbi_log)


# ### Does it work?

# Think about how to verify that your implementations of Viterbi (i.e. `compute_w`, `opt_path_prob`, `backtrack`, and there log-transformed variants `compute_w_log`, `opt_path_prob_log`, `backtrack_log`) are correct.
# 
# One thing that should hold is that the probability of a most likely path as computed by `opt_path_prob` (or `opt_path_prob_log`) for a given sequence of observables (e.g. `x_short` or `x_long`) should be equal to the joint probability of a corresponding most probable path as found by `backtrack` (or `backtrack_log`) and the given sequence of observables. Why?
# 
# Make an experiment that validates that this is the case for your implementations of Viterbi and `x_short` and `x_long`. You use your code from last week to compute the joint probability

# In[127]:


#Code from last week:
def joint_prob(model, x, z):
    """Calculates joint probability of a Hidden Markov Model.
    Input: model = model of type 'hmm'. x = sequence of observables indices. z = sequence of hidden states indices.
    Output: probability value"""
    
    prob = model.init_probs[z[0]] #initial probabilities: p(z_1)
    
    for i in range(1, len(z)):
        prob *= model.trans_probs[z[i-1]][z[i]]  #transition probs: *= p(z_n|z_n-1) from n = 2 to N
        
    for i in range(0, len(z)):
        prob *= model.emission_probs[z[i]][x[i]]  #emission probs: *= p(x_n|z_n) from n = 1 to N

    return prob

def joint_prob_log(model, x, z):
    """Calculates joint log probability of a Hidden Markov Model.
    Input: model = model of type 'hmm'. x = sequence of observables indices. z = sequence of hidden states indices.
    Output: log probability value"""
    
    prob = log(model.init_probs[z[0]])  #initial log probabilities: log(p(z_1))
    
    for i in range(1, len(z)):
        prob += log(model.trans_probs[z[i-1]][z[i]])  #transition log probs: += log(p(z_n|z_n-1)) from n = 2 to N
        
    for i in range(0, len(z)):
        prob += log(model.emission_probs[z[i]][x[i]])  #emission log probs: += log(p(x_n|z_n)) from n = 1 to N
    
    return prob


# In[171]:


####### hmm_7_state ###########

# Check that opt_path_prob is equal to joint_prob(hmm_7_state, x_short, z_viterbi)
w = compute_w(hmm_7_state, translate_observations_to_indices(x_short))
z_viterbi = backtrack(w, hmm_7_state, translate_observations_to_indices(x_short))
print(opt_path_prob(w) == joint_prob(hmm_7_state, translate_observations_to_indices(x_short), z_viterbi))

# Check that opt_path_prob_log is equal to joint_prob_log(hmm_7_state, x_short, z_viterbi_log)
w = compute_w_log(hmm_7_state, translate_observations_to_indices(x_short))
z_viterbi_log = backtrack_log(w, hmm_7_state, translate_observations_to_indices(x_short))
print(opt_path_prob_log(w) == 
      joint_prob_log(hmm_7_state, translate_observations_to_indices(x_short), z_viterbi_log))

# Check that opt_path_prob is equal to joint_prob(hmm_7_state, x_long, z_viterbi)
w = compute_w(hmm_7_state, translate_observations_to_indices(x_long))
z_viterbi = backtrack(w, hmm_7_state, translate_observations_to_indices(x_long))
print(round(opt_path_prob(w),10) == 
      round(joint_prob(hmm_7_state, translate_observations_to_indices(x_long), z_viterbi), 10))

# Check that opt_path_prob_log is equal to joint_prob_log(hmm_7_state, x_long, z_viterbi_log)
w = compute_w_log(hmm_7_state, translate_observations_to_indices(x_long))
z_viterbi_log = backtrack_log(w, hmm_7_state, translate_observations_to_indices(x_long))
print(round(opt_path_prob_log(w),10) == 
      round(joint_prob_log(hmm_7_state, translate_observations_to_indices(x_long), z_viterbi_log), 10))

####### hmm_3_state ###########

# Check that opt_path_prob is equal to joint_prob(hmm_3_state, x_short, z_viterbi)
w = compute_w(hmm_3_state, translate_observations_to_indices(x_short))
z_viterbi = backtrack(w, hmm_3_state, translate_observations_to_indices(x_short))
print(opt_path_prob(w) == 
      joint_prob(hmm_3_state, translate_observations_to_indices(x_short), z_viterbi))

# Check that opt_path_prob_log is equal to joint_prob_log(hmm_3_state, x_short, z_viterbi_log)
w = compute_w_log(hmm_3_state, translate_observations_to_indices(x_short))
z_viterbi_log = backtrack_log(w, hmm_3_state, translate_observations_to_indices(x_short))
print(opt_path_prob_log(w) == 
      joint_prob_log(hmm_3_state, translate_observations_to_indices(x_short), z_viterbi_log))

# Check that opt_path_prob is equal to joint_prob(hmm_3_state, x_long, z_viterbi)
w = compute_w(hmm_3_state, translate_observations_to_indices(x_long))
z_viterbi = backtrack(w, hmm_7_state, translate_observations_to_indices(x_long))
print(round(opt_path_prob(w),10) == 
      round(joint_prob(hmm_3_state, translate_observations_to_indices(x_long), z_viterbi), 10))

# Check that opt_path_prob_log is equal to joint_prob_log(hmm_3_state, x_long, z_viterbi_log)
w = compute_w_log(hmm_3_state, translate_observations_to_indices(x_long))
z_viterbi_log = backtrack_log(w, hmm_3_state, translate_observations_to_indices(x_long))
print(round(opt_path_prob_log(w), 10) == 
      round(joint_prob_log(hmm_3_state, translate_observations_to_indices(x_long), z_viterbi_log), 10))


# Do your implementations pass the above checks?
# 
# They pass the above checks as we obtain the same values except in the x_long test, where there is a difference on the last decimals. If we round this number up to 10 decimals we get that they are the same.

# ### Does log transformation matter?

# Make an experiment that investigates how long the input string can be before `backtrack` and `backtrack_log` start to disagree on a most likely path and its probability.

# In[133]:


for i in range(498, 504, 1):
    x = x_long[:i]
    
    x_trans = translate_observations_to_indices(x)

    print("i:", i)
    w = compute_w(hmm_3_state, x_trans)
    z_viterbi = backtrack(w, hmm_3_state, x_trans)
    
    w = compute_w_log(hmm_3_state, x_trans)
    z_viterbi_log = backtrack_log(w, hmm_3_state, x_trans)
    print("z_viterbi_log == z_viterbi ->", z_viterbi_log == z_viterbi)
    print("\n")


# In[134]:


for i in range(528, 533, 1):
    x = x_long[:i]
    
    x_trans = translate_observations_to_indices(x)

    print("i:", i)
    w = compute_w(hmm_7_state, x_trans)
    z_viterbi = backtrack(w, hmm_7_state, x_trans)
    
    w = compute_w_log(hmm_7_state, x_trans)
    z_viterbi_log = backtrack_log(w, hmm_7_state, x_trans)
    print("z_viterbi_log == z_viterbi ->", z_viterbi_log == z_viterbi)
    print("\n")


# ** Your answer here: **
# 
# For the 3-state model, `backtrack` and `backtrack_log` start to disagree on a most likely path and its probability
# for **i = 501 **.
# 
# For the 7-state model, `backtrack` and `backtrack_log` start to disagree on a most likely path and its probability
# for **i = 530 ** .
# 
# 
