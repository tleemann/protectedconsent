## Implementations of the resampling stategies for PUC.

import numpy as np
from numpy.random import multinomial, shuffle

def exhaustive_resampling(X, A, Y, return_original_indictor=False):
    """ Exhaustively Resample all combinations in the dataset. 
        X: (M, N) numpy matrix of the samples. The optional features are supposed to be in the last
            R < N columns
        A: (M, R) binary availability indicators.
        Y: (M) labels
        return_original_indictor: bool, if True, an additional binary array is returned indicating
            which record were from the original dataset.
        Return the adapted matrices, X, A, Y. Note that dataset size may increase exessively.
    """
    sample_list = []
    indicator_list = []
    result_len = 0 # Length of the final dataset as it is built up
    for j in range(len(X)):
        sample_list.append(_single_resample(X[j,:], A[j,:], Y[j]))
        result_len += len(sample_list[-1][0])
        indicator_list.append(result_len-1)
    dist_lists = list(zip(*sample_list))

    if return_original_indictor:
        indicator_array = np.zeros(result_len, dtype=int)
        indicator_array[indicator_list] = 1
        return np.vstack(dist_lists[0]), np.vstack(dist_lists[1]), np.concatenate(dist_lists[2]), indicator_array
    else:
        return np.vstack(dist_lists[0]), np.vstack(dist_lists[1]), np.concatenate(dist_lists[2])


def _single_resample(Xi, Mi, Yi):
    num_optional_avail = int(np.sum(Mi))
    num_smpl = 2 ** num_optional_avail # Number of samples after oversampling
    Y = np.ones(num_smpl)*Yi
    M = np.zeros((num_smpl, len(Mi)))
    M[:,Mi==1] = _create_binary_missingness_pattern(None, num_optional_avail)
    X = np.tile(Xi, (num_smpl, 1))
    X[:,-len(Mi):] = X[:,-len(Mi):]*M
    return X, M, Y


def _create_binary_missingness_pattern(m, run_left):
    if run_left == 0:
        return m if m is not None else np.empty((1,0), dtype=np.int)
    if m is None:
        m = np.array([0,1], dtype=np.int).reshape(-1,1)
    else:
        upper = np.hstack((np.ones((len(m),1)), m.copy()))
        lower = np.hstack((np.zeros((len(m),1)), m.copy()))
        m = np.vstack((lower, upper))
    return _create_binary_missingness_pattern(m, run_left-1)


def random_resampling(X, A, Y, n=10000):
    """ Randomly sample a batch of n observations.
        X: (M, N) numpy matrix of the samples. The optional features are supposed to be in the last
            R < N columns
        A: (M, R) binary availability indicators.
        Y: (M) labels
    """

    # Draw some indices according to the sample weights.
    sample_weights = np.sum(A, axis=1)
    num_smpl = np.power(2, sample_weights) # Number of samples after oversampling
    pvals = num_smpl/np.sum(num_smpl)
    draws = multinomial(n, pvals, size=1)
    #draws = num_smpl
    indx_set = np.repeat(np.arange(len(X)), draws.flatten()).flatten() # indices of drawn instances
    shuffle(indx_set)
    # Randomly drop available feature with p=0.5
    A_dropout = (np.random.rand(len(indx_set), A.shape[1]) > 0.5)

    A_new = A[indx_set]*A_dropout
    # Set corresponding features to zero in the samples.
    X_new = X[indx_set]
    X_new[:,-A.shape[1]:]= X_new[:,-A.shape[1]:]*A_new

    return X_new, A_new, Y[indx_set]




