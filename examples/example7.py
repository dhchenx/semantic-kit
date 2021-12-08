from semantickit.similarity.googlenews_similarity import googlenews_similarity
data_path= r'GoogleNews-vectors-negative300.bin'

sim=googlenews_similarity(data_path,'human','people')

print(sim)
