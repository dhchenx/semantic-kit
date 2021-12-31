from gensim import corpora, models, similarities
import gensim
import numpy as np

def load_model(data_path="model_dict.dict",corpus_path="model_corpus.mm"):
    dictionary = corpora.Dictionary.load(data_path)
    corpus = corpora.MmCorpus(corpus_path)  # comes from the first tutorial, "From strings to vectors"
    print(corpus)

    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=2)
    return lda,dictionary,corpus

def get_vec_lsi(data_path="model_dict.dict",corpus_path="model_corpus.mm",doc="The LDA algorithm was explained"):
    # doc = "The LDA algorithm was explained"
    lda,dictionary,corpus=load_model(data_path,corpus_path)
    vec_bow = dictionary.doc2bow(doc.lower().split())
    vec_lsi = lda[vec_bow]  # convert the query to LSI space
    print(vec_lsi)
    return vec_lsi

def get_vec_lsi_by_model(lda,dictionary,doc="The LDA algorithm was explained"):
    # doc = "The LDA algorithm was explained"
    vec_bow = dictionary.doc2bow(doc.lower().split())
    vec_lsi = lda[vec_bow]  # convert the query to LSI space
    print(vec_lsi)
    return vec_lsi

def save_index(lda,corpus,save_path='model_sentences.index'):
    index = similarities.MatrixSimilarity(lda[corpus])  # transform corpus to LSI space and index it
    index.save(save_path)

def load_index(index_path="model_sentences.index"):
    index = similarities.MatrixSimilarity.load(index_path)
    return index

def query(index,vec_lsi):
    sims = index[vec_lsi]  # perform a similarity query against the corpus
    print("unsorted", list(enumerate(sims)))  # print (document_number, document_similarity) 2-tuples
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    print("sorted", sims)  # print sorted (document number, similarity score) 2-tuples
    return sims

def get_sim_list(index,vec_lsi,sentences_file="sentences.txt"):
    sims = index[vec_lsi]
    documents = []
    with open(sentences_file, encoding="utf-8") as file:
        for line in file.readlines():
            documents.append(line.strip())
    # print("--------------Result---------------")
    list=[]
    for item in enumerate(sims):
        # print(item)
        # print(item[1][0], item[1][1], documents[item[1][0]])
        list.append([item[1][0], item[1][1], documents[item[1][0]]])
    return list

def get_similarity_between_sentences(lda,dictionary,sentence1="The LDA algorithm",sentence2="a new unseen document"):
    vec_bow1 = dictionary.doc2bow(sentence1.lower().split())
    vec_lda1 = lda[vec_bow1]
    vec_bow2 = dictionary.doc2bow(sentence2.lower().split())
    vec_lda2 = lda[vec_bow2]
    # Cosine similarity is universally useful & built-in:
    sim = gensim.matutils.cossim(vec_lda1, vec_lda2)
    # print("similarity between two sentences: ", sim)
    return sim


def get_helliger_distance(lda,dictionary,sentence1,sentence2):
    vec_bow1 = dictionary.doc2bow(sentence1.lower().split())
    vec_lda1 = lda[vec_bow1]
    vec_bow2 = dictionary.doc2bow(sentence2.lower().split())
    vec_lda2 = lda[vec_bow2]
    # Hellinger distance is useful for similarity between probability distributions (such as LDA topics):

    dense1 = gensim.matutils.sparse2full(vec_lda1, lda.num_topics)
    dense2 = gensim.matutils.sparse2full(vec_lda2, lda.num_topics)
    sim = np.sqrt(0.5 * ((np.sqrt(dense1) - np.sqrt(dense2)) ** 2).sum())
    # print("Helliger distance: ", sim)
    return sim

