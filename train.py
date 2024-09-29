import json
from nltk_utlis import tokenise, stem, bag

with open('intents.json', 'r') as f:
    intents = json.load(f)
    
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenise(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', ',','!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
