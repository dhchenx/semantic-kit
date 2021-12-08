# natural language processing
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
stop_words = stopwords.words('english')

#synset collection
from nltk.corpus import wordnet as wn

def WordNet_Hyponyms(syn_id):
    syn = wn.synset(syn_id)
    hypos = syn.hyponyms()
    slist=sorted([lemma.name() for synset in hypos for lemma in synset.lemmas()])
    unique_term=[]
    unique_term_count=[]
    for idx,x in enumerate(slist):
        if x not in unique_term:
            unique_term.append(x)
            unique_term_count.append(1)
        else:
            index=unique_term.index(x)
            unique_term_count[index]=unique_term_count[index]+1
    return unique_term,unique_term_count

def WordNet_Hypernyms(syn_id):
    syn = wn.synset(syn_id)
    hyper = syn.hypernyms()
    slist=sorted([lemma.name() for synset in hyper for lemma in synset.lemmas()])
    unique_term=[]
    unique_term_count=[]
    for idx,x in enumerate(slist):
        if x not in unique_term:
            unique_term.append(x)
            unique_term_count.append(1)
        else:
            index=unique_term.index(x)
            unique_term_count[index]=unique_term_count[index]+1
    return unique_term,unique_term_count

#tokenize the definition string to meaningful word list
def Tokenize(str,unique=True):
    tok = nltk.word_tokenize(str)
    tok=[word.lower() for word in tok]
    # remove stopwords
    tok = [word for word in tok if word not in stop_words]
    #print(tok)
    #lemmatize
    lemmatizer = WordNetLemmatizer()
    ltok = [lemmatizer.lemmatize(w, pos='n') for w in tok]
    #remove pos tags
    tagged = nltk.pos_tag(ltok)
    term = []
    for idx, w in enumerate(ltok):
        if tagged[idx][1][0] in ['N', 'V', 'J']:
            term.append(w)
    if not unique:
        return term
    unique_term=[]
    for w in term:
        if w not in unique_term:
            unique_term.append(w)
    return unique_term

def WordNet_Synonym(syn_id):
    s_lemma=wn.synset(syn_id).lemma_names()
    print("lemma: ",s_lemma)
    s_definition=wn.synset(syn_id).definition()
    s_example=" ".join(wn.synset(syn_id).examples())
    print("definition: ", s_definition)
    print("example: ",s_example)
    all_str=(s_definition+" "+s_example).lower()
    all_str_list=all_str.split()
    com_words=[]
    for i in range(0,len(all_str_list)-1):
        t=(all_str_list[i]+"_"+all_str_list[i+1]).lower()
        #print(t)
        if t in s_lemma:
            all_str=all_str.replace(all_str_list[i]+" "+all_str_list[i+1], t)
            com_words.append(t)
    all_str_tok=Tokenize(all_str,unique=False)
    all_str_tok.extend(com_words)
    all_str_tok.extend(s_lemma)
    #print(all_str)
    unique_term=[]
    unique_term_count=[]
    for idx,x in enumerate(all_str_tok):
        if x not in unique_term:
            unique_term.append(x)
            unique_term_count.append(1)
        else:
            index=unique_term.index(x)
            unique_term_count[index]=unique_term_count[index]+1
    return unique_term,unique_term_count

