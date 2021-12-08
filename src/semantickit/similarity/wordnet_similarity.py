import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

def wordnet_similarity_all(synset1,synset2):
    brown_ic = wordnet_ic.ic('ic-brown.dat')
    semcor_ic = wordnet_ic.ic('ic-semcor.dat')

    a = wn.synset(synset1)
    b = wn.synset(synset2)

    dict_result={}

    # path-based
    sim_path = a.path_similarity(b)
    sim_wup = a.wup_similarity(b)
    sim_lch = a.lch_similarity(b)
    print("sim_path", sim_path)
    print("sim_wup", sim_wup)
    print("sim_lch", sim_lch)
    dict_result["sim_path"]=sim_path
    dict_result["sim_wup"]=sim_wup
    dict_result["sim_lch"]=sim_lch

    # mutual information-based
    sim_res = a.res_similarity(b, brown_ic)
    sim_jcn = a.jcn_similarity(b, brown_ic)
    sim_lin = a.lin_similarity(b, semcor_ic)
    print("sim_res", sim_res)
    print("sim_jcn", sim_jcn)
    print("sim_lin", sim_lin)
    dict_result["sim_res"]=sim_res
    dict_result["sim_jcn"]=sim_jcn
    dict_result["sim_lin"]=sim_lin

    return dict_result



