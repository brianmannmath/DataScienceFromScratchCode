#A simple univariate regression model

from __future__ import division
from functools import partial

import GradientDescent as gd
import numpy as np

class SimpleLinearRegression(object):

    def __init__(self): 
        self.beta = None
        self.alpha = None

    def train(self, x, y):
        #Data is a list of pairs x, y
        beta = sum( y_i*(x_i - np.mean(x)) for y_i, x_i in zip(x,y) )/sum( (x_i - np.mean(x))**2 for x_i in x )
        alpha = np.mean(y) - beta*np.mean(x)
        self.beta, self.alpha = beta, alpha

    def predict(self, x):
        return self.beta*x + self.alpha

    def parameters(self):
        return self.alpha, self.beta

    def _sum_of_squares(self, y):
        return sum( y_i**2 for y_i in y )

    def r_squared(self, x, y):
        return 1.0 - self._sum_of_squares( self.beta*x_i + self.alpha - y_i for x_i, y_i in zip(x, y) ) / self._sum_of_squares( y_i - np.mean(y) for y_i in y ) 
