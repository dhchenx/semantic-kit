# Semantic Similarity and Relatedness Toolkit

A toolkit to estimate semantic similarity and relatedness between two words/sentences. 

## Installation
```pip
pip install semantic-kit
```

## Functions
1. Lesk algorithm and improved version
2. Similarity algorithms including WordNet , word2vec similarity, LDA, and googlenews-based methods
3. Distance algorithms like jaccard, soren, levenshtein, and their improved versions

## Examples
### Lesk Algorithm
```python
from semantickit.relatedness.lesk import lesk
from semantickit.relatedness.lesk_max_overlap import lesk_max_overlap
sent = ['I', 'went', 'to', 'the', 'bank', 'to', 'deposit', 'money', '.']
m1, s1 = lesk(sent, 'bank', 'n')
m2, s2 = lesk_max_overlap(sent, 'bank', 'n')
print(m1,s1)
print(m2,s2)
```
### WordNet-based Similarity
```python
from semantickit.similarity.wordnet_similarity import wordnet_similarity_all
print(wordnet_similarity_all("dog.n.1","cat.n.1"))
```

### Corpus-based Similarity
```python
from semantickit.similarity.word2vec_similarity import build_model,similarity_model
# build similarity model based on text source
build_model(data_path="text8",save_path="wiki_model")
# estimate similarity between words using the built model
sim=similarity_model("wiki_model","france","spain")
# print result
print("word2vec similarity: ",sim)
```

### Pre-trained model-based Similarity
```python
from semantickit.similarity.googlenews_similarity import googlenews_similarity
data_path= r'GoogleNews-vectors-negative300.bin'
sim=googlenews_similarity(data_path,'human','people')
print(sim)
```

## Weighted Levenshtein
```python
from semantickit.distance.n_gram.train_ngram import TrainNgram
from semantickit.distance.weighted_levenshtein import weighted_levenshtein,Build_TFIDF

# train model
train_data_path = 'wlev/icd10_train.txt'
wordict_path = 'wlev/word_dict.model'
transdict_path = 'wlev/trans_dict.model'
words_path="wlev/dict_words.txt"
trainer = TrainNgram()
trainer.train(train_data_path, wordict_path, transdict_path)

# build words tf-idf file
Build_TFIDF(train_data_path,words_path)

# estimate weight lev distance
s0='颈结缔组织良性肿瘤'
s1='耳软骨良性肿瘤'
result=weighted_levenshtein(s0,s1, word_dict_path=wordict_path,trans_dict_path=transdict_path,data_path=train_data_path,words_path=words_path)
print(result)
```

### LDA similarity
```python
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
```

## License
The `Semantic-Kit` project is provided by [Donghua Chen](https://github.com/dhchenx). 

