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
    def __init__(self, v, n, s, train_file_name, test_file_name):
        self.vocabulary_type = v
        self.corpus = None
        self.corpus_size = 0
        self.n_gram_type = n
        self.n_gram_eu = None
        self.n_gram_ca = None
        self.n_gram_gl = None
        self.n_gram_es = None
        self.n_gram_en = None
        self.n_gram_pt = None
        self.smoothing = s
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name

        self.generateVocabulary()
        self.generateNgram()

    def getCorpus(self):
        return self.corpus

    def getNGramEN(self):
        return self.n_gram_en

    def getTrainingFile(self):
        return self.train_file_name

    def getTestingFile(self):
        return self.test_file_name

    def generateNgram(self):
        if self.n_gram_type == 1:
            self.n_gram_eu = np.zeros(shape=(1, self.corpus_size), dtype=float)
            self.n_gram_ca = np.zeros(shape=(1, self.corpus_size), dtype=float)
            self.n_gram_gl = np.zeros(shape=(1, self.corpus_size), dtype=float)
            self.n_gram_es = np.zeros(shape=(1, self.corpus_size), dtype=float)
            self.n_gram_en = np.zeros(shape=(1, self.corpus_size), dtype=float)
            self.n_gram_pt = np.zeros(shape=(1, self.corpus_size), dtype=float)

        elif self.n_gram_type == 2:
            self.n_gram_eu = np.zeros(shape=(1, self.corpus_size, self.corpus_size), dtype=float)
            self.n_gram_ca = np.zeros(shape=(1, self.corpus_size, self.corpus_size), dtype=float)
            self.n_gram_gl = np.zeros(shape=(1, self.corpus_size, self.corpus_size), dtype=float)
            self.n_gram_es = np.zeros(shape=(1, self.corpus_size, self.corpus_size), dtype=float)
            self.n_gram_en = np.zeros(shape=(1, self.corpus_size, self.corpus_size), dtype=float)
            self.n_gram_pt = np.zeros(shape=(1, self.corpus_size, self.corpus_size), dtype=float)

        elif self.n_gram_type == 3:
            self.n_gram_eu = np.zeros(shape=(1, self.corpus_size, self.corpus_size, self.corpus_size), dtype=float)
            self.n_gram_ca = np.zeros(shape=(1, self.corpus_size, self.corpus_size, self.corpus_size), dtype=float)
            self.n_gram_gl = np.zeros(shape=(1, self.corpus_size, self.corpus_size, self.corpus_size), dtype=float)
            self.n_gram_es = np.zeros(shape=(1, self.corpus_size, self.corpus_size, self.corpus_size), dtype=float)
            self.n_gram_en = np.zeros(shape=(1, self.corpus_size, self.corpus_size, self.corpus_size), dtype=float)
            self.n_gram_pt = np.zeros(shape=(1, self.corpus_size, self.corpus_size, self.corpus_size), dtype=float)


    def generateVocabulary(self):
        if self.vocabulary_type == 0:
            self.corpus = list(string.ascii_lowercase)
            self.corpus_size = len(self.corpus)

        elif self.vocabulary_type == 1:
            self.corpus = list(string.ascii_letters)
            self.corpus_size = len(self.corpus)

        # MUST USE isalpha() method
        elif self.vocabulary_type == 2:
            self.corpus = list(string.ascii_letters)
            self.corpus_size = len(self.corpus)


    def buildGram(self, data):
        grams = []
        if self.n_gram_type == 1:
            for char in data:
                if (char in self.corpus):
                    grams.append(char)

        elif self.n_gram_type == 2:
            for i, char in enumerate(data):
                if i < (len(data) - 1):
                    if (char in self.corpus) and (data[i+1] in self.corpus):
                        grams.append(char)
                else:
                    pass

        # elif self.n_gram_type == 3:
        return grams

    def getCharIndex(self, char):
        try:
            return self.corpus.index(char)
        except ValueError as err:
            print(err)

    def train(self):
        with open(self.train_file_name) as f:
            tweets = f.readlines()

        for tweet in tweets:
            print(tweet)

            elements = tweet.split()

            # Get all info from a tweet
            userId = elements[0]
            username = elements[1]
            language = elements[2]
            data = ' '.join(elements[3:])

            grams = self.buildGram(data)
            print(grams)

            # clean data
            # print(np.where(self.n_gram_eu == ))


            # if language == 'eu':


            break


def main():
    print("Naive main.")

if __name__ == '__main__':
    main()
