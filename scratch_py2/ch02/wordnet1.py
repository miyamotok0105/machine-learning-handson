import nltk
from nltk.corpus import wordnet as wn

# nltk.download('wordnet')
print(wn.synsets('motorcar'))

print(wn.synset('car.n.01').lemma_names())

print(wn.synsets('printer'))

