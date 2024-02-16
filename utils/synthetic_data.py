## Some utility functions to generate synthetic data
import numpy as np
import pandas as pd

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sample_binary_pmatrix(n_features):
    """ Sample a matrix with probabilites for a binary feature.
        Each column corresponds to a class
        Return (n_features, 2) array.
    """
    return np.random.rand(n_features, 2)

def create_multiple_sigmoidal_missingness(df, columns, alpha, imputation=0):
    df_copy = df.copy(deep=True)
    pd.set_option('mode.chained_assignment', None) #prevent unnecessary warning
    for column_name, a_use in zip(columns, alpha):
        mean_val = df_copy[column_name].mean()
        print("Mean of", column_name, mean_val)
        p_missing = sigmoid(a_use*(df_copy[column_name].values-mean_val))
        missingness = np.random.rand(len(df_copy)) < p_missing
        df_copy[column_name + " missing"] = pd.Series(missingness)
        df_copy[column_name].iloc[df_copy[column_name + " missing"]==1] = imputation
    return df_copy

def sample_binary_dataset(p_matrix, p_y1, n_records=100):
    """
        Sample a synthetic binary dataset with two classes.
        p_matrix: feature probabilities [n_features, 2] matrix
            [i, 0] = p(x_i=1|y=1), [i, 1] = p(x_i=1|y=0)
        p_y1: Prior for class 
        return X (feature matrix), Y (labels)
    """
    labels = np.random.rand(n_records)  < p_y1 # binary labels.
    dataset = np.random.rand(n_records, len(p_matrix))
    comparer = p_matrix[:,labels.astype(int)].T
    print(comparer.shape)
    return (dataset < comparer).astype(int), labels

def induce_multiple_independent_missingness(X, Y, f_missingness, p_missing_y0, p_missing_y1):
    """ 
        Randomly remove multiple features (indices f_missingness)
        return X (with missing features last), M (missingness matrix)
    """
    binary_missingness = np.ones((X.shape[1],), dtype=bool)
    binary_missingness[f_missingness] = 0

    X_base = X[:, binary_missingness]
    X_optional = X[:, ~binary_missingness]
    print(X_optional.shape)
    missingmatrix = np.zeros((len(X), len(f_missingness)))
    missingmatrix[Y==0] = np.random.rand(len(Y[Y==0]), len(f_missingness)) < p_missing_y0.reshape(1,-1)
    missingmatrix[Y==1] = np.random.rand(len(Y[Y==1]), len(f_missingness)) < p_missing_y1.reshape(1,-1)
    return np.hstack((X_base, X_optional*(1-missingmatrix))), missingmatrix

def binary_ground_truth_off(X, prior_y1, p_matrix, p_missing, off=True):
    """ Compute ground truth off predictions for sample from 
        the binary dataset provided by sample_binary_dataset and induce_multiple_independent_missingness
        p_matrix [N, 2] probability of feature val 1 if y=0, y=1
        p_missing [M, 2] probability of feature missing if y=0, y=1
        off: If true, the possiblity under optional feature fairness (OFF) is returned, else the probability w/o fairness constraints
        is computed.
    """
    #1. compute log odds for P(Y=1)
    odds_x1 = np.log(p_matrix[:,1]/p_matrix[:,0]) # odds if feature = 1
    odds_x0 = np.log((1-p_matrix[:,1])/(1-p_matrix[:,0])) # odds if feature = 0
    # Add odds for availability.
    odds_avail = np.log((1-p_missing[:,1])/(1-p_missing[:,0]))
    odds_unavail = np.log((p_missing[:,1])/(p_missing[:,0]))
    odds_x1[-len(odds_avail):] += odds_avail
    odds_x0[-len(odds_avail):] += odds_avail
    
    
    Xodds = np.zeros(X.shape)
    Xodds += (X==0)*odds_x0.reshape(1,-1)
    Xodds += (X==1)*odds_x1.reshape(1,-1)
    if not off:
        Xodds[:, -len(odds_unavail):] += (X[:, -len(odds_unavail):]==-1)*odds_unavail.reshape(1,-1)

    #print(Xodds[:10])
    total_odds = np.sum(Xodds, axis=1) + np.log(prior_y1/(1.0-prior_y1))
    return np.exp(total_odds)/(1.+np.exp(total_odds))


def sample_continuous_dataset(base_weights, base_offset, n_features, n_records = 100, feature_mean = None, feature_cov = None):
    if feature_cov is None:
        feature_cov = np.eye(n_features)
    if feature_mean is None: 
        feature_mean = np.zeros(n_features)
    pts = np.random.multivariate_normal(mean=feature_mean, cov=feature_cov, size=n_records)
    logits = np.sum(base_weights.reshape(1, -1)*pts, axis=1) + base_offset
    predict_proba = sigmoid(logits)
    labels = np.random.rand(n_records) < predict_proba
    return pts, labels

def sample_missing_features(X, y, n_features, ceil_pres, U, V, lambda_, mu):
    """ ceil_pres: (R) matrix: overall amount of presence [0, 1]. Values of 0.1 indicates a ~ 10 percent chance
        for the feature to be present (when both classes are equally likely, no offset.)
        V: (N, R) matrix with the dependency between zi and B
        U: (N, R) matrix specifying the dependency between missingness and b
        lambda_: (R) matrix specifying the difference in odds for the two classes y=0, y=1
        mu: (2, R), the means of the normal distribution.
    """
    ret_array = []
    m_array = []
    for i in range(n_features):
        vib = np.matmul(X, V[:, i].reshape(-1,1)).flatten() # vi^T*b
        uib = np.matmul(X, U[:, i].reshape(-1,1)).flatten()
        mi_prob = np.exp((2*y-1)*(uib+lambda_[i])) # (odds = +- u^tb+lambda_i)
        opp_prob = 1.0
        # The larger odds than one, p=1. Else p= odds because other prob = 1.
        max_prob = (mi_prob>=opp_prob)*mi_prob + (mi_prob<opp_prob)*opp_prob

        #mi_prob = (mi_prob)/(1.0 + mi_prob)
        mi_prob = ceil_pres[i]*(mi_prob/max_prob)
        mi = np.random.rand(len(y))<mi_prob
        zi = np.random.randn(len(y)) +  vib + mu[y.astype(int), i]
        m_array.append(mi.reshape(-1, 1))
        ret_array.append(mi.reshape(-1, 1)*zi.reshape(-1, 1))
    return np.hstack(ret_array), np.hstack(m_array)

    


