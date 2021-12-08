from gensim import corpora, models, similarities

dictionary = corpora.Dictionary.load('dictionary.dict')
corpus = corpora.MmCorpus("corpus.mm")
lda = models.LdaModel.load("model.lda") #result from running online lda (training)

index = similarities.MatrixSimilarity(lda[corpus])
index.save("simIndex.index")

docname = "docs/the_doc.txt"
doc = open(docname, 'r').read()
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lda = lda[vec_bow]

sims = index[vec_lda]
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims)

