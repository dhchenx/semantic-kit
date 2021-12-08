from nltk.corpus import wordnet

def lesk_max_overlap(context_sentence, ambiguous_word, pos=None, synsets=None):
    context = set(context_sentence)
    if synsets is None:
        synsets = wordnet.synsets(ambiguous_word)

    if pos:
        synsets = [ss for ss in synsets if str(ss.pos()) == pos]

    if not synsets:
        return None

    max_overlap = 0
    max_idx = 0
    for idx, ss in enumerate(synsets):

        ss_def = ss.definition().split()
        #print(idx,ss_def)
        inter_s = context.intersection(ss_def)
        #print(inter_s)
        if len(inter_s) >= max_overlap:
            max_overlap = len(inter_s)
            max_idx = idx

    return max_overlap,synsets[max_idx]
