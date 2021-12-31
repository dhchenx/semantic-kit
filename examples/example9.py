from semantickit.app.lsi_model import build_lsi_model
from semantickit.app.lda_similarity import load_model,save_index,load_index,get_similarity_between_sentences,get_helliger_distance

# data file
data_path="lsi_data/sentences.txt"
# stopwords file
stop_words_path="lsi_data/stopwords_english.txt"
# build lsi model
build_lsi_model(data_path=data_path,stopwords_path=stop_words_path,save_dict_path="model.dict",save_corpus_path="model.corpus")
# create lda model and dictionary
lda,dictionary,corpus=load_model(data_path="model.dict",corpus_path="model.corpus")
# save index file
save_index(lda,corpus,save_path="model.index")
# load the saved index file
index=load_index("model.index")
# get the similarity
similarity=get_similarity_between_sentences(lda,dictionary,"The LDA algorithm","a new unseen document")
print("similarity: ", similarity)
# get the distance
helliger_distance=get_helliger_distance(lda,dictionary,"The LDA algorithm","a new unseen document")
print("helliger distance: ",helliger_distance)