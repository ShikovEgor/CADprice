import numpy as np


def weighted_normalized_jaccard(set1, set2):
    """
    Calculates the weighted normalized Jaccard index between two sets.
    
    Arguments:
    set1 -- a set of elements (ground truth)
    set2 -- a set of elements (predicted)
    
    Returns:
    The weighted normalized Jaccard index between the two sets.
    """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0
    weight = len(set1) / len(set2) if len(set2) > len(set1) else len(set2) / len(set1)
    return jaccard * weight

def get_mf(graph, truth, labels):
    mfs = [set() for _ in range(len(truth))]
    def check_node(graph, start_truth, i, labels):
        target_label = labels[start_truth]
        mfs[i].update([start_truth])
        for k in graph.neighbors(start_truth):
            if labels[k] == target_label:
                if k not in mfs[i]:
                    mfs[i].update([start_truth])
                    check_node(graph, k, i, labels)
    
    for i, start_truth in enumerate(truth):
        check_node(graph, start_truth, i, labels)
    max_measure = -1
    max_index = -1
    for i, mf in enumerate(mfs):
        measure = weighted_normalized_jaccard(set(truth), mf)
        if measure > max_measure:
            max_measure = measure
            max_index = i
    return mfs[max_index], max_measure


def get_mf_jaccard(sample, labels):
    graph = sample['graph'].to_networkx()
    mfs = {}
    for truth in sample['mechanical_features']:
        mfs[tuple(truth)] = get_mf(graph, truth, labels)
    return np.mean([x[1] for x in mfs.values()])