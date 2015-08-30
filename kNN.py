from collections import Counter
from math import sqrt

import numpy as np

def majority_vote(labels):
    #Assume labels are sorted nearest to furthest

    counts = Counter(labels)
    winners = counts.most_common()
    if len(winners) == 1:
        return winners[0][0]
    else:
        majority_vote(labels[:-1])

def mag_squared(v):
    return np.dot(v,v)

def dist(x, y):
    return sqrt(mag_squared(np.subtract(x,y)))

def knn(k, labeled_points, new_point):
    #labeled_points are a list of pairs points, labels
    sorted_by_distance = sorted(labeled_points, key=lambda (p, _): dist(p, new_point))
    closest_k = sorted_by_distance[:k]
    labels = [label for _, label in closest_k]
    return majority_vote(labels)
