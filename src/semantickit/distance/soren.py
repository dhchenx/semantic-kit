from similarity.sorensen_dice import SorensenDice

def soren(word1,word2,k=1):
    soren = SorensenDice(k=1)
    sim = soren.similarity(word1,word2)
    return sim