import stanfordnlp
nlp = stanfordnlp.Pipeline(processors='tokenize', lang='hi')
doc = nlp("text")
f = open('tokenized.txt',mode = 'w',encoding='utf-8')
for i, sentence in enumerate(doc.sentences):
    for token in sentence.tokens:
        f.write(f"{token.text}\n")
f.close()
