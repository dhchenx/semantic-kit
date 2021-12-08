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