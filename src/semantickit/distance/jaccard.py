def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    print(intersection)
    print(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection / union)
