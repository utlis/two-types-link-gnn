import re
from gensim.models import FastText

def tokenize(term):
    return re.findall(r"\b\w+\b", term.lower())

def build_vocab(terms):
    vocab = {}
    index = 0
    for term in terms:
        tokens = tokenize(term)
        for token in tokens:
            if token not in vocab:
                vocab[token] = index
                index += 1
    return vocab

def load_fasttext_model(model_path):
    return FastText.load_fasttext_format(model_path)

