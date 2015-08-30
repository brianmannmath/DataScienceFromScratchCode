#Code for Naive Bayes Classifier
from math import exp, log
from __future__ import division

def tokenize(message):
    message = message.lower()
    words = re.findall("[a-z0-9]", message)
    return(set(words))

def count_words(training_set):
    counts = dict()
    for message, spam in training_set:
        for word in tokenize(message):
            if spam:
                if word in counts:
                    counts[word][0] += 1
                else:
                    counts[word] = [1,0]
            else:
                if word in counts:
                    counts[word][1] += 1
                else:
                    counts[word] = [0,1]
    return counts

def word_probabilities(counts, total_spams, total_non_spams, k=0.5):
    #Returns a list of tuples w, P(w, spam), P(w, ~spam)
    return [(w,
            (spam + k)/(total_spams + 2*k),
            (not_spam + k)/(total_non_spams + 2*k)) for w, (spam, not_spams) in counts.iteritems()]

def spam_probability(word_probs, message):
    words = tokenize(message)
    log_prob_spam, log_prob_not_spam = 0, 0  
    for word, spam, not_spam in word_probs:
        if word in words:
            log_prob_spam += log(spam)
            log_prob_not_spam += log(not_spam)
        else:
            log_prob_spam += log(1-spam)
            log_prob_not_spam += log(1-not_spam)
    return exp(log_prob_spam)/(exp(log_prob_spam) + exp(log_prob_not_spam))

def NaiveBayesClassifier(object):
    
    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []

    def train(self, training_set):
        total_spams = sum( 1 for _, spam in training_set if spam )
        total_not_spams = len(training_set) - total_spams
        self.word_probs = word_probabilities(count_words(training_set), total_spams, total_not_spams, self.k)

    def classify(self, message):
        return spam_probability(self.word_probs, message)


        
