from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec

def build_model(data_path,save_path=None):
    sentences = word2vec.Text8Corpus("text8")
    model = word2vec.Word2Vec(sentences)
    if save_path!=None:
        model.save(save_path)
    return model

def similarity_model(load_path,word1,word2):
    model = Word2Vec.load(load_path)
    sim = model.similarity(word1, word2)
    return sim





