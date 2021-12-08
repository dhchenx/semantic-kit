from semantickit.relatedness.lesk import lesk

# context
sentence="I went to the bank to deposit my money"

# word
word="bank"

a = lesk(word,sentence)

print("\n\nSynset:",a)
if a is not None:
    print("Meaning:",a.definition())
    num=0
    print("\nExamples:")
    for i in a.examples():
        num=num+1
        print(str(num)+'.'+')',i)