import numpy as np
import random

##################
#    SAMPLING    #
##################

def sample_r(n : int) -> np.array:
    """
    Samples a vector r of size <n> from the standard normal distribution.
        Parameters:
            n (int) : dimension of the vector
        Returns:
            samples (np.array) : the sampled vector
    """
    samples = np.random.standard_normal(n)
    return samples

##################
#      MCMC      #
##################

def compute_V(s : np.array) -> float:
    """
    Computes the V energy function of a vector s of 1's and -1's (Ising model/spins).
        Parameters:
            s (np.array) : vector to compute energy of
        Returns:
            V (float) : energy of vector
    """
    V = 0
    for i in range(len(s) - 1): 
        V += (s[i] - s[i + 1]) ** 2
    return 0.5 * V

def init_S(n : int) -> np.array:
    """
    Initializes a random vector of 1's and -1's selected uniformly (Ising model/spins).
        Parameters:
            n (int) : dimension of vector
        Returns:
            initial_s (np.array) : random vector of dimension n
    """
    initial_s = np.random.rand(n)
    for i in range(n): 
        if (initial_s[i] < 0.5):
            initial_s[i] = -1
        else:
            initial_s[i] = 1
    return initial_s

def random_walk(s : np.array) -> np.array:
    """
    Performs one step of a random walk on the state space of {-1, 1}^n.
    Flips the spin of one particle and returns a copy.
        Parameters: 
            s (np.array) : current state vector
        Returns:
            s_prime (np.array) : new state vector with randomly flipped spin on one particle
    """
    n = len(s)
    s_prime = s.copy()
    flip = random.randint(0, n - 1)
    s_prime[flip] *= -1
    return s_prime

def run_mcmc(samples : int, r : np.array, beta : float) -> np.array:
    """
    Runs an MCMC scheme to sample from a Gibbs Distribution over the state space {-1, 1}^n.
    Uses compute_V as the energy function.
    Computes the P_beta or random matrix based on these samples.
        Parameters:
            samples (int) : number of samples to take
            r (int) : baseline r samples to use when creating P_beta
            beta (int) : beta hyperparameter
        Returns:
            P_beta (np.array) : P_beta after sampling <samples> from the Gibbs Distribution using MCMC
    """
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
    return P_beta
        
##################
#   COVARIANCE   #
##################
def create_P0(r : np.array) -> np.array:
    """
    Creates P_0, or P when beta = 0.
    Simply just a diagonal matrix with <r>^2 on the diagonal (square computed element-wise).
        Parameters:
            r (np.array) : vector to square and put on diagonal
        Returns:
            diag (np.array) : diagonal matrix with r^2 on the diagonal
    """
    return np.diag(r ** 2, k=0)

#################
#   SAMPLING    #
#################
def sample_cov(cov : np.array, samples : int) -> np.array:
    """
    Samples from a multivariate normal distribution with covariance given by <cov>.
        Parameters:
            cov (np.array) : covariance matrix to use
            samples (int) : number of samples to take
        Returns:
            rv (np.array) : randomly sampled vectors
    """
    n = len(cov)
    rv = np.random.multivariate_normal(np.zeros(n), cov, samples)
    return rv