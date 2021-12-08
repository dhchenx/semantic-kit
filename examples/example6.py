from semantickit.similarity.word2vec_similarity import build_model,similarity_model

# build similarity model based on text source
build_model(data_path="text8",save_path="wiki_model")

# estimate similarity between words using the built model
sim=similarity_model("wiki_model","france","spain")

# print result
print("word2vec similarity: ",sim)