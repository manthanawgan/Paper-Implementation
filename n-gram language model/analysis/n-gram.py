import numpy as np
from nltk.util import ngrams
from preprocess import get_tokenized_sentences

class NgramCounter:
    def __init__(self, text_file : str) -> None:
        self.sentence_generator = get_tokenized_sentences(text_file)
        self.count()

