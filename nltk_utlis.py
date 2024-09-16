import nltk
nltk.download('punkt_tab')

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenise(sentence: str) -> list[str]:
    return nltk.word_tokenize(sentence)

def stem(word: str):
    return stemmer.stem(word.lower())

def bag(tokenised_sentence, words):
    pass
