import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora

def build_lsi_model(data_path="sentences.txt",stopwords_path="stopwords_english.txt",save_dict_path="model_dict.dict",save_corpus_path="model_corpus.mm"):
    documents = []

    with open(data_path, encoding="utf-8") as file:
        documents = [l.strip() for l in file]

    stoplist = []

    with open(stopwords_path, encoding="utf-8") as file:
        stoplist = [l.strip() for l in file]

    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in documents]

    # remove words that appear only once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]

    from pprint import pprint  # pretty-printer
    pprint(texts)

    dictionary = corpora.Dictionary(texts)
    dictionary.save(save_dict_path)  # store the dictionary, for future reference
    print(dictionary)

    print(dictionary.token2id)

    # new_doc = "Human computer interaction"
    # new_vec = dictionary.doc2bow(new_doc.lower().split())
    # print(new_vec)  # the word "interaction" does not appear in the dictionary and is ignored

    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize(save_corpus_path, corpus)  # store to disk, for later use
    print(corpus)

# build_lsi_model()
