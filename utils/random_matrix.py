import numpy as np
from matplotlib import pyplot as plt

############################
# RANDOM MATRIX GENERATION #
############################

def random_matrix_goe(n : int) -> np.array:
    """
    Generates a random matrix drawn from the Gaussian Orthogonal Ensemble (GOE).
        Parameters:
            n (int) : dimension of matrix
        Returns:
            X (np.array): n by n GOE matrix
    """
    X = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                X[i][j] = np.random.normal(0, 1/n)
            else:
                X[i][j] = np.random.normal(0, 1/(2*n))
                X[j][i] = np.random.normal(0, 1/(2*n))
    return X

def random_matrix_wishart(n : int, m: int) -> np.array:
    """
    Generates a random matrix drawn from the Wishart Ensemble.
        Parameters:
            n (int) : dimension of matrix
            m (int) : number of columns in intermediate rectangular matrix used in Wishart generation
        Returns:
            P (np.array): n by n Wishart matrix
    """
    X = np.random.normal(0, 1, (n, m))
    P = (1/n) * (X @ X.transpose())
    return P

###########################
#        PLOTTING         #
###########################

def plot_avg_GOE(n : int, iters : int, color : str) -> None:
    """
    Plots the CDF of the spectra of <iters> randomly drawn <n> by <n> GOE matrices.
        Parameters:
            n (int) : dimension of matrices
            iters (int) : number of samples
            color (str) : color of line to plot CDF
        Returns: None
    """
    all_vals = []
    for _ in range(iters):
        X = random_matrix_goe(n)
        all_vals.extend(np.linalg.eigvalsh(X))
    
    np_all_vals = np.sort(np.array(all_vals))
    y_vals = np.array(range(n * iters))/float(n * iters)
    label = "n = " + str(n)
    plt.plot(np_all_vals, y_vals, color=color, label=label)
    
def plot_avg_wishart(n : int, m : int, iters : int, color : str) -> None:
    """
    Plots the CDF of the spectra of <iters> randomly drawn <n> by <n> Wishart matrices.
        Parameters: 
            n (int) : dimension of matrices
            m (int) : intermediate number of columns used in Wishart generation.
            iters (int) : number of samples
            color (str) : color of line to plot CDF
        Returns: None
    """
    all_vals = []
    for _ in range(iters):
        X = random_matrix_wishart(n, m)
        all_vals.extend(np.linalg.eigvalsh(X))

    np_all_vals = np.sort(np.array(all_vals))
    y_vals = np.array(range(n * iters))/float(n * iters)
    ratio = round(m/n, 3)
    label = "m/n = " + str(ratio)
    plt.plot(np_all_vals, y_vals, color=color, label=label)
