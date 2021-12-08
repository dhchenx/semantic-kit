from gensim.models.word2vec import Word2Vec
import gensim

def googlenews_similarity(data_path,word1,word2):
    model = gensim.models.KeyedVectors.load_word2vec_format(
       data_path,
        binary=True)
    sim = model.similarity(word1,word2)

    print(sim)
    return sim