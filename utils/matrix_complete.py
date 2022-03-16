import numpy as np
import random

##################
#    SAMPLING    #
##################
def sample_r(n : int) -> np.array:
    samples = np.random.standard_normal(n)
    return samples

##################
#      MCMC      #
##################
def compute_V(s : np.array) -> float:
    V = 0
    for i in range(len(s) - 1): 
        V += (s[i] - s[i + 1]) ** 2
    return 0.5 * V

def init_S(n : int) -> np.array:
    initial_s = np.random.rand(n)
    for i in range(n): 
        if (initial_s[i] < 0.5):
            initial_s[i] = -1
        else:
            initial_s[i] = 1
    return initial_s

def random_walk(s : np.array) -> np.array:
    n = len(s)
    s_prime = s.copy()
    flip = random.randint(0, n - 1)
    s_prime[flip] *= -1
    return s_prime

def run_mcmc(samples : int, r : np.array, beta : float) -> np.array:
    n = len(r)
    P_beta = np.zeros((n, n))

    s_cur = init_S(n)
    v_cur = compute_V(s_cur)
    for _ in range(samples):
        s_new = random_walk(s_cur)    
        v_new = compute_V(s_new)
        prob_accept = np.exp(-beta * max(v_new - v_cur, 0))

        if random.random() < prob_accept:
            s_cur = s_new
            v_cur = v_new
        
        rs = np.multiply(r, s_cur)
        to_add = np.outer(rs, rs)
        P_beta += to_add
    
    P_beta /= samples

    converted = list(s_cur)
    print(converted.count(1))
    print(converted.count(-1))
    return P_beta
        
##################
#   COVARIANCE   #
##################
def create_P0(diag : np.array) -> np.array:
    return np.diag(diag ** 2, k=0)

