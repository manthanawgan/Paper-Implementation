import numpy as np
from nltk.util import ngrams
from preprocess import get_tokenized_sentences

class NgramCounter:
    def __init__(self, text_file : str) -> None:
        self.sentence_generator = get_tokenized_sentences(text_file)
        self.count()

    def count(self):
        #count num of n-grams in text (both overall and starting text), one sentence/sequence at a time
        self.sentence_count = 0
        self.token_count = 0
        self.counts = {}
        self.start_counts = {}

        for sentence in self.sentence_generator:
            self.sentence_count += 1
            self.token_count += len(sentence)

            for ngram_length in range(1, 6):
                for ngram_position, ngram in enumerate(ngrams(sentence, ngram_length)):
                    if ngram_position == 0:
                        self.start_counts[ngram] = self.start_counts.get(ngram, 0) + 1
                    self.counts[ngram] = self.counts.get(ngram, 0) + 1                

class NgramModel:
    def __init__(self, train_counter: NgramCounter) -> None:
        