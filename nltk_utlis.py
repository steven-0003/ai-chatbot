import nltk
nltk.download('punkt_tab')

import numpy as np

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenise(sentence: str) -> list[str]:
    return nltk.word_tokenize(sentence)

def stem(word: str):
    return stemmer.stem(word.lower())

def bag(tokenised_sentence, words):
    tokenised_sentence = [stem(w) for w in tokenised_sentence]
    b = np.zeros(len(words), dtype=np.float32)

    for i, w in enumerate(words):
        if w in tokenised_sentence:
            b[i] = 1.0

    return b
