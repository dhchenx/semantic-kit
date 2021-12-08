
from semantickit.relatedness.lesk import lesk
from semantickit.relatedness.lesk_max_overlap import lesk_max_overlap

sent = ['I', 'went', 'to', 'the', 'bank', 'to', 'deposit', 'money', '.']

m1, s1 = lesk(sent, 'bank', 'n')
m2, s2 = lesk_max_overlap(sent, 'bank', 'n')

print(m1,s1)
print(m2,s2)

