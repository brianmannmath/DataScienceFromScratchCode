#Logistic regression model

import math
import GradientDescent as gd
import numpy as np

def logistic(x):
    return 1.0/(1 + math.exp(-x))

def logistic_prime(x):
    return logisitic(x) * (1 -logistic(x))

def logistic_log_likelihood(x_i, y_i, beta):
#p(y_i | x_i, B) = logistic(x_iB)^y_i*(1-logistic(x_iB)^(1-y_i)
    if y_i == 1:
        return math.log(logistic(np.dot(x_i, beta)))
    else:
        return math.log(1-logistic(np.dot(x_i, beta)))

def logistic_log_likelihood(x, y, beta):
    return sum( logistic_log_likelihood(x_i, y_i, beta)
                for x_i, y_i in zip(x, y) )

