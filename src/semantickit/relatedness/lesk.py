from nltk.corpus import wordnet

def lesk(context_sentence, ambiguous_word, pos=None, synsets=None):

    context = set(context_sentence)
    if synsets is None:
        synsets = wordnet.synsets(ambiguous_word)

    if pos:
        synsets = [ss for ss in synsets if str(ss.pos()) == pos]

    if not synsets:
        return None

    max_len, sense = max(
        (len(context.intersection(ss.definition().split())), ss) for ss in synsets
    )

    return max_len,sense



