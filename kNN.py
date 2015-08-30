from collections import Counter

def majority_vote(labels):
    #Assume labels are sorted nearest to furthest

    counts = Counter(labels)
    winners = counts.most_common()
    if len(winners) == 1:
        return winners[0][0]
    else:
        majority_vote(labels[:-1])
