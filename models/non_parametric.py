import numpy as np

class NonParametricRegression():
    """ Implementation of the Non-Parametric Regressor described in the paper. """
    def __init__(self):
        self.sum_dict = {}
        self.cnt_dict = {}

    def fit(self, X, Y):
        for i in range(len(X)): # Naive implementaion
            key = tuple(X[i])
            if key in self.sum_dict:
                self.sum_dict[key] += int(Y[i])
                self.cnt_dict[key] += 1
            else:
                self.sum_dict[key] = int(Y[i])
                self.cnt_dict[key] = 1

    def predict(self, X):
        """ Standard regression function.
            See the paper draft for a derivation.
        """
        Ypred = np.zeros(len(X))
        for i in range(len(X)): # Naive implementaion
            key = tuple(X[i])
            Ypred[i] = (self.sum_dict[key]/(self.cnt_dict[key]+1.0)) if key in self.sum_dict else 0
        return Ypred

    def predict_proba(self, X):
        py1 = self.predict(X)
        return np.stack((1.0-py1, py1)).T
