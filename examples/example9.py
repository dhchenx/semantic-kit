# use wordnet to generate multi-lang keywords
from semantickit.lang.wordnet import *
if __name__=="__main__":
    text = "digitalization meets carbon neutrality, digital economy"
    nltk.download("wordnet")
    nltk.download('omw')
    dict_lang_all=get_all_related_word_from_text(text)
    print()
    for lang in dict_lang_all:
        print(lang, ','.join(dict_lang_all[lang]))