#Decision Tree classifier

import numpy as np
import math

from collections import Counter
from functools import partial

def entropy(class_probabilities):
    return sum( -p*math.log(p,2) for p in class_probabilities if p)

#Data is of the form features, labels
def class_probabilities(labels):
    return [ count/len(labels) for count in Counter(labels).values() ]

def data_entropy(data):
    labels = [label for _, label in data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)

def partition_entropy(subsets):
    total = sum( len(subset) for subset in length(subsets) )
    return sum( data_entropy(subset)*len(subset)/total for subset in subsets )

def partition_by(inputs, attribute):
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][attribute]
        groups[key].append(input)

    return groups

def partition_entropy_by(inputs, attribute):
    partition = partition_by(inputs, attribute)
    return partition_entropy(partition.values())

class Tree(object):

    def build_tree_id3(inputs, split_candidates=None):
        if split_candidates == None:
            split_candidates = inputs[0][0].keys()

        num_inputs = len(inputs)
        num_trues = sum( 1 for _, label in inputs if label )
        num_falses = num_inputs - num_trues

        if num_trues == 0:
            return False
        if num_falses == 0:
            return True

        if not split_candidates:
            return num_trues >= num_falses

        best_attribute = min(split_candidates, key=partial(partition_entropy_by, inputs))
        
        partitions = partition_by(inputs, attribute)
        new_candidates = [candidate for candidate in split_candidates if candidate != best_attribute]
        
        subtrees = { attribute_value : build_tree_id3(subset, new_candidates) for attribute_value, subset in partitions.iteritems() }
        subtrees[None] = num_trues >= num_falses

        return (best_attribute, subtrees)


    def classify(tree, input):
        #tree is either True, False, or a pair (attribute, value:tree dict)
        if tree in [True, False]:
            return tree

        attribute, subtree_dict = tree
        subtree_key = input.get(attribute)

        if subtree_key not in subtree_dict:
            subtree_key = None

        return classify(subtree_dict[subtree_key], input)
        

