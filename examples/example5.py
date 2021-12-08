from nltk.corpus import wordnet as wn
from semantickit.similarity.wordnet_hyper_hypo import WordNet_Hypernyms,WordNet_Hyponyms,WordNet_Synonym

# hyponyms
motorcar = wn.synset('car.n.01')
types_of_motorcar = motorcar.hyponyms()
print(types_of_motorcar)
print(sorted([lemma.name() for synset in types_of_motorcar for lemma in synset.lemmas()]))

# hypernyms
motorcar.hypernyms()
paths=motorcar.hypernym_paths()
print(motorcar.root_hypernyms())

# call functions
print(WordNet_Hyponyms('car.n.01'))
print(WordNet_Hypernyms('car.n.01'))
print(WordNet_Synonym('voice.n.01'))

# descriptions
print("synsets: ",wn.synsets('motorcar'))
print("lemma names: ",wn.synset('car.n.01').lemma_names())
print("definition: ",wn.synset('car.n.01').definition())  #定义
print("lemmas: ", wn.synset('car.n.01').lemmas())
print("examples: ",wn.synset('car.n.01').examples())
