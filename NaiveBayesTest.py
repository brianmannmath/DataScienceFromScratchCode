from __future__ import division
from NaiveBayes import NaiveBayesClassifier
from collections import Counter

import glob, re, random

path = r"/home/ubuntu/Projects/DataScienceFromScrach/DataScienceFromScratchData/*/*"

data = []
for fn in glob.glob(path):
    is_spam = 'ham' not in fn
    with open(fn, 'r') as file:
        for line in file:
            if line.startswith("Subject:"):
                subject = re.sub(r"^Subject:", "", line).strip()
                data.append((subject, is_spam))

def split_data(data, p):
    return data[:int(len(data)*p)], data[int(len(data)*p):]

def in_random_order(data):
    indices = [i for i, _ in enumerate(data)]
    random.shuffle(indices)
    result = []
    for i in indices:
        result.append(data[i])
    return result

random.seed(0)
train_data, test_data = split_data(in_random_order(data), 0.75)
classifier = NaiveBayesClassifier()
classifier.train(train_data)

classified = [(message, is_spam, classifier.classify(message)) for message, is_spam in test_data]

counts = Counter( (is_spam, spam_prob > 0.5) for (_, is_spam, spam_prob) in classified )
