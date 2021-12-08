from nltk.wsd import lesk

sent = 'I went to the bank to deposit my money'
ambiguous = 'bank'
print(lesk(sent, ambiguous))
print(lesk(sent, ambiguous).definition())
