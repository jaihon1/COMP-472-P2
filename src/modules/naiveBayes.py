### Vocabulary
# v = 0 -> Fold the corpus to lowercase and use only the 26 letters of the alphabet [a-z]
# v = 1 -> Distinguish up and low cases and use only the 26 letters of the alphabet [a-z, A-Z]
# v = 2 -> Distinguish up and low cases and use all characters accepted by the built-in isalpha() method

### N-gram
# n = 0 -> character unigrams
# n = 1 -> character bigrams
# n = 2 -> character trigrams

### Smoothing
# s = {0,...,1}

import string
import numpy as np

class NaiveBayes():
    def __init__(self, v, n, s, training_file_name, testing_file_name):
        self.vocabulary_type = v
        self.corpus = None
        self.n_gram_type = n
        self.n_gram = None
        self.smoothing = s
        self.training_file_name = training_file_name
        self.testing_file_name = testing_file_name

        self.generateVocabulary()

    def getCorpus(self):
        return self.corpus

    def getNGram(self):
        return self.n_gram

    def getTrainingFile(self):
        return self.training_file_name

    def getTestingFile(self):
        return self.testing_file_name


    def generateVocabulary(self):
        if self.vocabulary_type == 0:
            self.corpus = list(string.ascii_lowercase)

        elif self.vocabulary_type == 1:
            self.corpus = list(string.ascii_uppercase)

        elif self.vocabulary_type == 2:
            self.corpus = list(string.ascii_letters)


def main():
    print("Naive main.")

if __name__ == '__main__':
    main()
