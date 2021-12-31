from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

def get_words_with_all_langs(synset_id):
    langs = sorted(wn.langs())
    list_lang = []
    list_useful_lang=[]
    for lang in langs:
        if lang not in list_lang:
            list_lang.append(lang)
    dict_lang={}
    for lang in list_lang:
        try:
            names = wn.synset(synset_id).lemma_names(lang)
            # print(lang, names)
            if len(names)!=0:
                dict_lang[lang]=names
                if lang not in list_useful_lang:
                    list_useful_lang.append(lang)
        except:
            # print("Error: Lang: "+lang)
            pass
    return dict_lang,list_useful_lang

def get_words_from_sentence(sentence):
    word_list=word_tokenize(sentence)
    filtered_words = [word.lower() for word in word_list if word not in stopwords.words('english')]
    # st = LancasterStemmer()
    lem = WordNetLemmatizer()
    list_w=[]
    for w in filtered_words:
        list_w.append(lem.lemmatize(w))
    return list_w

def get_hyponyms(synset_name):
    synsets=wn.synset(synset_name).hyponyms()
    dict_lang_all = {}
    for synset in synsets:
        synset_name = synset.name()
        dict_langs, list_useful_lang = get_words_with_all_langs(synset_name)
        for lang in list_useful_lang:
            if lang in dict_lang_all:
                for ww in dict_langs[lang]:
                    dict_lang_all[lang].append(ww)
            else:
                dict_lang_all[lang] = []
                for ww in dict_langs[lang]:
                    dict_lang_all[lang].append(ww)
    return dict_lang_all

def get_hypernyms(synset_name):
    synsets=wn.synset(synset_name).hypernyms()
    dict_lang_all = {}
    for synset in synsets:
        synset_name = synset.name()
        dict_langs, list_useful_lang = get_words_with_all_langs(synset_name)
        for lang in list_useful_lang:
            if lang in dict_lang_all:
                for ww in dict_langs[lang]:
                    dict_lang_all[lang].append(ww)
            else:
                dict_lang_all[lang] = []
                for ww in dict_langs[lang]:
                    dict_lang_all[lang].append(ww)
    return dict_lang_all

def get_all_related_word_from_text(text,use_hyponym=False,use_hypernym=False):

    word_list = get_words_from_sentence(text)
    print(word_list)
    dict_lang_all = {}
    for w in word_list:
        print(w)
        synsets = wn.synsets(w)
        print(synsets)
        for synset in synsets:
            synset_name = synset.name()
            print("synset.name = ", synset_name)
            dict_langs, list_useful_lang = get_words_with_all_langs(synset_name)
            print(dict_langs)
            # current
            for lang in list_useful_lang:
                if lang in dict_lang_all:
                    for ww in dict_langs[lang]:
                        dict_lang_all[lang].append(ww)
                else:
                    dict_lang_all[lang] = []
                    for ww in dict_langs[lang]:
                        dict_lang_all[lang].append(ww)
            # hyponyms
            if use_hyponym:
                dict_lang_hyponyms=get_hyponyms(synset_name)
                for lang in dict_lang_hyponyms:
                    if lang in dict_lang_all:
                        for ww in dict_lang_hyponyms[lang]:
                            dict_lang_all[lang].append(ww)
                    else:
                        dict_lang_all[lang] = []
                        for ww in dict_lang_hyponyms[lang]:
                            dict_lang_all[lang].append(ww)
            # hypernyms
            if use_hypernym:
                dict_lang_hypernyms = get_hypernyms(synset_name)
                for lang in dict_lang_hypernyms:
                    if lang in dict_lang_all:
                        for ww in dict_lang_hypernyms[lang]:
                            dict_lang_all[lang].append(ww)
                    else:
                        dict_lang_all[lang] = []
                        for ww in dict_lang_hypernyms[lang]:
                            dict_lang_all[lang].append(ww)

        # print()
    return dict_lang_all


