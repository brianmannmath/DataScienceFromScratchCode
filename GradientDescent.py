'''
Functions from the gradient descent chaper of Data Science From Scratch
'''

from __future__ import division
from functools import partial
from numpy import random

#1-dimensional derivative

def difference_quotient(f, x, h):
    return (f(x+h) - f(x))/h

#Multivariate functions from R^n -> R

def partial_difference_quotient(f, v, i, h):
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
    return (f(w) - f(v))/h

def estimate_gradient(f, v, h):
    return [partial_difference_quotient(f, v, i, h) for i, _ in enumerate(v)]

def step(f, grad, start, alpha):
    return [v - alpha*w for v, w in zip(start, grad(start))]

def safe(f):
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')
    return safe_f

def minimize_batch(f, grad, start, tolerance=0.0000001):
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    
    theta = start
    value = f(theta)
    while True:
        thetas = [step(f, grad, theta, step_size) for step_size in step_sizes]
        theta = min(thetas, key=f)
        if(abs(f(theta) - value) < tolerance):
            return theta
        else:
            value = f(theta)
       
def negate(f):
    def f_neg(*args, **kwargs):
        return -f(*args, **kwargs)
    return f_neg

def negate_all(f):
    def f_neg_all(*args, **kwargs):
        return [-y for y in f(*args, **kwargs)]
    return f_neg_all

#Stochastic Gradient Descent
def in_random_order(data):
    indices = [i for i, _ in enumerate(data)]
    random.shuffle(indices)
    for i in indices:
        yield data[i]

def minimize_stochastic(f, grad, x, y, theta, alpha=0.01):
    data = zip(x, y)
    min_theta, min_value = None, float('inf')
    iterations_with_no_improvement = 0

    while iterations_with_no_improvement < 100:
        value = sum( f(x_i, y_i, theta) for x_i, y_i in data )
        if value < min_value:
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
        else:
            iterations_with_no_improvement += 1
            alpha *= 0.9

        for x_i, y_i in in_random_order(data):
            theta = [v - alpha*w for v, w in zip(theta, grad(x_i, y_i, theta))]

    return min_theta

