# Logistic regression models supporting fair treatment of missing features.

from sklearn.linear_model import LogisticRegression
import numpy as np
import math
from utils import sigmoid

class BaseFairLogisticRegression():
    def predict_proba(self, X, M):
        """ Fair prediction function. 
            See the paper draft for a derivation.
            :param X: (M, n_basefeatures+n_missingfeatures) data matrix (continuous)
            :param M: (M, n_missingfeatures) binary matrix indicating missingness (1 = present, 0 = missing)
        """

        # Compute log odds.
        X[:,self.n_basefeatures:] = X[:,self.n_basefeatures:] * M
        coefs, intercepts = self._compute_coefs(X, M)
        logits = np.sum(coefs*X, axis=1) + intercepts.flatten() # M
        return np.vstack((sigmoid(-logits), sigmoid(logits))).T

    def predict(self, X, M):
        """ Fair prediction function. 
            See the paper draft for a derivation.
            :param X: (M, n_basefeatures+n_missingfeatures) data matrix (continuous)
            :param M: (M, n_missingfeatures) binary matrix indicating missingness (1 = present, 0 = missing)
        """
        return self.predict_proba(X,M)[:,1] > 0.5

    def _compute_coefs(X, M):
        raise NotImplementedError("_compute_coefs needs to be implemented for every subclass")


class ContinousLogisticRegression(BaseFairLogisticRegression):
    """ A Logistic regression model, that supports continous
        features several missing features under the assumptions detailed in the paper.
    """
    def __init__(self, n_basefeatures, n_missingfeatures, random_state=42):
        self.n_basefeatures = n_basefeatures # N
        self.n_missingfeatures = n_missingfeatures # R
        # Init parameters
        self.w = 0 # [N] -vector
        self.t = 0 # scalar
        self.omega = np.zeros((self.n_basefeatures, self.n_missingfeatures))
        self.beta = np.zeros(self.n_missingfeatures)
        self.s = np.zeros(self.n_missingfeatures)
        self.random_state = random_state

    def fit(self, X, M, Y):
        """ Fit the model with O(n_basefeatures*n_missing_features) parameters. 
            :param X: (M, n_basefeatures+n_missingfeatures) data matrix (continuous)
                The first n_basefeatures features are treated as base features that are always present. 
                The last n_missingfeatures can also be missing.
            :param M:
                (M, n_missingfeatures) binary matrix indicating missingness (1 = present, 0 = missing)
                Only the present values will be used to estimate the model, i.e., it does not matter which
                value is used to impute missing features.
        """
        # First fit the base model.
        Xbase = X[:, :self.n_basefeatures]
        base_model = LogisticRegression(solver='liblinear', random_state=self.random_state)
        base_model.fit(Xbase, Y)
        self.w = base_model.coef_.flatten()
        self.t = base_model.intercept_.flatten()

        # Second fit other models
        for i in range(self.n_missingfeatures):
            presence = (M[:, i] > 0)
            ext_data = np.hstack((Xbase[presence,:], X[presence, self.n_basefeatures+i].reshape(-1,1)))
            #print(ext_data.shape)
            feature_model =  LogisticRegression(solver='liblinear', random_state=self.random_state)
            feature_model.fit(ext_data, Y[presence])
            # compute parameters
            self.omega[:, i] = feature_model.coef_[0, :-1] - self.w
            self.beta[i] = feature_model.coef_[0, -1]
            self.s[i] = feature_model.intercept_.flatten() - self.t

    def _compute_coefs(self, X, M):
        """ Compute coefficients for each point in X.
            Return X.shape coefficient mattrix and len(X) offset vecotr.
        """
        # Compute log odds.
        Xbase = X[:, :self.n_basefeatures].reshape(-1, self.n_basefeatures)
        Xoptional = X[:, self.n_basefeatures:].reshape(-1, self.n_missingfeatures)

        #print(Xbase.shape, Xoptional.shape, M.shape)
        Xoptional = Xoptional * M # Make sure Xoptional is zero imputed.
        
        # Compute w + \sum_i m_i*omega_i
        base_weights = self.w.reshape(1,-1) + np.matmul(M, self.omega.T) # (M, n_basefeatures)

        #optional_weights stay constant at self.beta with 0 imputation.

        offsets = self.t + np.matmul(M, self.s.reshape(-1, 1))
        return np.hstack((base_weights, np.broadcast_to(self.beta.reshape(1, -1), (len(X), len(self.beta))))), offsets

class NonFairContinousLogisticRegression():
    """ A ground truth only model that fulfils no fairness constraints. """
    def __init__(self, base_gt, u, lambd, overall_pres):
        self.base_model = base_gt
        self.u = u
        self.lambd = lambd
        self.overall_pres = overall_pres
    
    def predict_proba(self, X, M):
        X[:,self.base_model.n_basefeatures:] = X[:,self.base_model.n_basefeatures:] * M

        uib = np.matmul(X[:, :self.base_model.n_basefeatures], self.u)
        mi_prob = uib+self.lambd.reshape(1,-1) 
        # Bigger prob.
        nb = self.overall_pres.reshape(1,-1) /sigmoid(np.abs(mi_prob))
        #print(nb[:10])
        sigmoid_all = sigmoid(mi_prob)
        odds_add = (1.0-nb*sigmoid_all)/(1.0-nb*(1.0-sigmoid_all)) #(n_missing_features, N)
        odds_add = np.log(odds_add)
        odds_add*=(1.0-M) # Times missingness (all not missing are 0)
        # From predict_proba
        #print(odds_add[:10])
        coefs, intercepts = self.base_model._compute_coefs(X, M)
        logits = np.sum(coefs*X, axis=1) + intercepts.flatten() + np.sum(odds_add, axis=1)
        return np.vstack((sigmoid(-logits), sigmoid(logits))).T




class BinaryNBLogisticRegression(BaseFairLogisticRegression):
    """ A Logistic regression model, that supports binary
        features with several missing features under naive bayes assumption detailed in the paper.
    """
    def __init__(self, n_basefeatures, n_missingfeatures):
        self.n_basefeatures = n_basefeatures # N
        self.n_missingfeatures = n_missingfeatures # R
        # Init parameters
        self.w = 0 # [N] -vector base model weights
        self.t = 0 # scalar base model intercept
        self.omega = np.zeros((self.n_basefeatures, self.n_missingfeatures))
        self.beta = np.zeros(self.n_missingfeatures)
        self.s = np.zeros(self.n_missingfeatures)

    def fit(self, X, M, Y):
        """ Fit the model with O(n_basefeatures*n_missing_features) parameters. 
            :param X: (M, n_basefeatures+n_missingfeatures) data matrix (binary)
                The first n_basefeatures features are treated as base features that are always present. 
                The last n_missingfeatures can also be missing.
            :param M:
                (M, n_missingfeatures) binary matrix indicating missingness (1 = present, 0 = missing)
                Only the present values will be used to estimate the model, i.e., it does not matter which
                value is used to impute missing features.
        """
        # First fit the base model.
        Xbase = X[:, :self.n_basefeatures]
        base_model = LogisticRegression(penalty="none")
        base_model.fit(Xbase, Y)
        self.w = base_model.coef_.flatten()
        self.t = base_model.intercept_.flatten()

        print(Y.astype(float).mean())
        class_ratio_offset = np.log(Y.astype(float).mean()/(1-Y.astype(float).mean()))
        print(class_ratio_offset)
        # Second fit other models
        for i in range(self.n_missingfeatures):
            presence = (M[:, i] > 0)
            # under the naive base assumption, we can fit each feature on its own.
            #ext_data = np.vstack((Xbase[presence,:], X[presence, n_basefeatures+i].reshape(-1,1)))
            #print(ext_data.shape)
            feature_model =  LogisticRegression(penalty="none")
            feature_model.fit(X[presence, self.n_basefeatures+i].reshape(-1,1), Y[presence])
            # compute parameters
            print(feature_model.coef_, feature_model.intercept_)
            self.beta[i] = feature_model.coef_[0, 0]
            self.s[i] = feature_model.intercept_.flatten() - class_ratio_offset
        
    def _compute_coefs(self, X, M):
        """ Compute coefficients for each point in X.
            Return X.shape coefficient mattrix and len(X) offset vecotr.
        """
        # Compute log odds.
        base_weights = self.w.reshape(1,-1)
        all_weights = np.hstack((base_weights,self.beta.reshape(1, -1)))
        #optional_weights stay constant at self.beta with 0 imputation.

        offsets = self.t + np.matmul(M, self.s.reshape(-1, 1)).flatten()
        return all_weights, offsets