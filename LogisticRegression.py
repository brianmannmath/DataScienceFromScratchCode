#Logistic regression model

from __future__ import division

import math
import GradientDescent as gd
import numpy as np

from functools import partial

def logistic(x):
    return 1.0/(1 + np.exp(-x))

def logistic_prime(x):
    return logisitic(x) * (1 -logistic(x))

def logistic_log_likelihood_i(x_i, y_i, beta):
    #p(y_i | x_i, B) = logistic(x_iB)^y_i*(1-logistic(x_iB)^(1-y_i)
    if y_i == 1:
        return np.log(logistic(np.dot(x_i, beta)))
    else:
        return np.log(1-logistic(np.dot(x_i, beta)))

def logistic_log_likelihood(x, y, beta):
    return sum( logistic_log_likelihood_i(x_i, y_i, beta)
                for (x_i, y_i) in zip(x, y) )

def log_likelihood_partial(x, y, beta, j):
    return sum( (y_i - logistic(np.dot(x_i, beta)))*x_i[j] for x_i, y_i in zip(x, y) )

def gradient(x, y, beta):
    return [log_likelihood_partial(x, y, beta, j) for j, _ in enumerate(beta)]

class LogisticRegression(object):

    def __init__(self):
        self.beta = None

    def train(self, X, y):
        X = [[1] + x for x in X]
        self.beta = gd.maximize_batch(partial(logistic_log_likelihood, X, y), partial(gradient, X, y), start=[1.0 for _ in X[0]]) 
