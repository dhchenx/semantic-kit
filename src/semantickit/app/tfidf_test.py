from gensim import corpora, models, similarities

dictionary = corpora.Dictionary.load('model_dict.dict')
corpus = corpora.MmCorpus('model_corpus.mm') # comes from the first tutorial, "From strings to vectors"
print(corpus)

tfidf = models.TfidfModel(corpus)

doc = "The LDA algorithm was explained"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = tfidf[vec_bow] # convert the query to LSI space
print(vec_lsi)

index = similarities.MatrixSimilarity(tfidf[corpus]) # transform corpus to LSI space and index it

#index.save('model_sentences.index')
#index = similarities.MatrixSimilarity.load('model_sentences.index')

sims = index[vec_lsi] # perform a similarity query against the corpus
print("unsorted",list(enumerate(sims))) # print (document_number, document_similarity) 2-tuples

sims = sorted(enumerate(sims), key=lambda item: -item[1])
print("sorted",sims) # print sorted (document number, similarity score) 2-tuples

documents=[]
with open("sentences.txt", encoding="utf-8") as file:
    for line in file.readlines():
        documents.append(line.strip())
print("--------------Result---------------")
for item in enumerate(sims):
    #print(item)
    print(item[1][0], item[1][1],documents[item[1][0]])
