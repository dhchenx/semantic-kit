from semantickit.distance.n_gram.max_ngram import MaxProbCut
from similarity.string_distance import StringDistance
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

class CharacterInsDelInterface:

    def deletion_cost(self, c):
        raise NotImplementedError()

    def insertion_cost(self, c):
        raise NotImplementedError()

def Normalization(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]

def Build_TFIDF(data_path='n_gram/data/icd10_train.txt',dict_path='dict_words.txt'):
    fo = open(data_path, mode='r', encoding='utf-8')
    corpus = []

    line = fo.readline()
    counter=0
    part_txt=''
    while line:
        line=line.replace('\n','')
        part_txt=part_txt+" "+line
        counter=counter+1
        if counter>=250:
            corpus.append(part_txt.strip())
            counter=0
            part_txt=''
        line = fo.readline()
    fo.close()

    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    unique_word_tfidf_list=[]
    unique_word_list=[]
    for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        #print(u"-------这里输出第", i, u"类文本的词语tf-idf权重------")
        for j in range(len(word)):
            if not(word[j] in unique_word_list):
                unique_word_list.append(word[j])
                unique_word_tfidf_list.append(weight[i][j])
            else:
                idx=unique_word_list.index(word[j])
                if unique_word_tfidf_list[idx]<weight[i][j]:
                    unique_word_tfidf_list[idx]=weight[i][j]
            #print(word[j], weight[i][j])

    unique_word_tfidf_list=Normalization(unique_word_tfidf_list)

    fo_dict = open(dict_path, 'w', encoding='utf-8')
    for idx,d in enumerate(unique_word_list):
        fo_dict.write(d+"\t"+str(unique_word_tfidf_list[idx])+"\n")
    fo_dict.close()
    return unique_word_list,unique_word_tfidf_list

class CharacterSubstitutionInterface:

    def cost(self, c0, c1):
        raise NotImplementedError()

class WeightedLevenshtein(StringDistance):

    def __init__(self, character_substitution, character_ins_del=None):
        self.character_ins_del = character_ins_del
        if character_substitution is None:
            raise TypeError("Argument character_substitution is NoneType.")
        self.character_substitution = character_substitution

    def distance(self, s0, s1):
        if s0 is None:
            raise TypeError("Argument s0 is NoneType.")
        if s1 is None:
            raise TypeError("Argument s1 is NoneType.")
        if s0 == s1:
            return 0.0
        if len(s0) == 0:
            return len(s1)
        if len(s1) == 0:
            return len(s0)

        v0, v1 = [0.0] * (len(s1) + 1), [0.0] * (len(s1) + 1)
        #用n-gram修改初始权重

        #print(v0,v1)

        v0[0] = 0
        for i in range(1, len(v0)):
            v0[i] = v0[i - 1] + self._insertion_cost(s1[i - 1])

        for i in range(len(s0)):
            s1i = s0[i]
            deletion_cost = self._deletion_cost(s1i)
            v1[0] = v0[0] + deletion_cost

            for j in range(len(s1)):
                s2j = s1[j]
                # print(i,j,s1i, ' -> ',s2j)
                cost = 0
                if s1i != s2j:
                    cost = self.character_substitution.cost(s1i, s2j)
                r_cost=0
                if s1i==s2j:
                    r_cost = self._relevant_cost(s1i)

                insertion_cost = self._insertion_cost(s2j)
                v1[j + 1] = min(v1[j] + insertion_cost, v0[j + 1] + deletion_cost, v0[j] + cost+r_cost)
                # print(v1[j+1])
            v0, v1 = v1, v0
        #print("v0",v0)
        return v0[len(s1)]

    #增强关键词的权重
    def _relevant_cost(self,c):
        if c=='肝':
            return 0
        return 0

    def _insertion_cost(self, c):
        if self.character_ins_del is None:
            return 1.0
        return self.character_ins_del.insertion_cost(c)

    def _deletion_cost(self, c):
        if self.character_ins_del is None:
            return 1.0
        return self.character_ins_del.deletion_cost(c)

#对于两个词的替代，增加其语义概率
# 如果两个词之间相似，那么替换的成本会很低
from similarity.metric_lcs import MetricLCS
from similarity.jarowinkler import JaroWinkler

class CharacterSubstitution(CharacterSubstitutionInterface):

    def mcls(self,c0,c1):
        metric_lcs = MetricLCS()
        dist_mlcs = metric_lcs.distance(c0, c1)
        # print(dist_mlcs)
        if dist_mlcs <= 0.3:
            return 0
        else:
            return dist_mlcs

    def jar(self,c0,c1):
        jarowinkler = JaroWinkler()
        sim_jar = jarowinkler.similarity(c0, c1)
        # print(sim_jar)
        if sim_jar >0.8:
            return 0
        else:
            return 1-sim_jar

    def cost(self, c0, c1):
        # print("calc sim: ",c0,c1)
        sim0=wn.synsets(c0,lang='cmn')
        #print(sim0)
        sim1=wn.synsets(c1,lang='cmn')
        #print(sim1)
        if len(sim0)==0 or len(sim1)==0:
            dist=self.jar(c0,c1)
            # print("dist",dist)
            return dist

        sim=sim0[0].path_similarity(sim1[0])

        if sim==None:
            dist= self.jar(c0,c1)
            # print("dist",dist)
            return dist

        if sim>0.01:sim=1
        # print("wn_similarity: ",sim)

        return 1-sim
        #if c0=='萎缩' and c1=='坏死':
        #   return 0.1
        #return 1.0

def FindInWordList(unique_word_list,unique_word_tfidf_list, word):
    if word in unique_word_list:
        idx=unique_word_list.index(word)
        return unique_word_tfidf_list[idx]
    else:
        return 0

#如果这个词很重要，删除的成本会很高
#如果这个词很重要，插入成本成本会是负数，很低 [-1, 1]
class CharacterIns(CharacterInsDelInterface):

    def __init__(self,data_path='n_gram/data/icd10_train.txt',dict_path='dict_words.txt'):
        self.w,self.f=Build_TFIDF(data_path,dict_path)

    def deletion_cost(self, c):
        tf_idf=FindInWordList(self.w, self.f,c)
        # print("delete check: ", c,tf_idf )
        return tf_idf

    def insertion_cost(self, c):
        tf_idf = FindInWordList(self.w, self.f, c)
        # print("delete check: ", c, tf_idf )
        return tf_idf

#build tf_idf
#Build_TFIDF()

def weighted_levenshtein(word1,word2,word_dict_path='n_gram/model/word_dict.model',trans_dict_path='n_gram/model/trans_dict.model',data_path="n_gram/data/icd10_train.txt",words_path="dict_words.txt"):
    cuter = MaxProbCut(word_dict_path,trans_dict_path )

    weighted_levenshtein = WeightedLevenshtein(CharacterSubstitution(), character_ins_del=CharacterIns(data_path,words_path))

    s0 = cuter.cut(word1)
    s1 = cuter.cut(word2)
    # print(s0)
    # print(s1)

    dist_wlev = weighted_levenshtein.distance(s0, s1)
    # print('lev_dist', dist_wlev)

    max_len = max(len(s0), len(s1))
    nor_lev = dist_wlev * 1.0 / max_len
    # print('normalized distance: ', nor_lev)
    # print('similarity: ', 1 - nor_lev)
    return {"lev_dist":dist_wlev,"nor_lev":nor_lev,"similarity":1-nor_lev}



