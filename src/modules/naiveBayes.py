### Vocabulary
# v = 0 -> Fold the corpus to lowercase and use only the 26 letters of the alphabet [a-z]
# v = 1 -> Distinguish up and low cases and use only the 26 letters of the alphabet [a-z, A-Z]
# v = 2 -> Distinguish up and low cases and use all characters accepted by the built-in isalpha() method

### N-gram
# n = 1 -> character unigrams
# n = 2 -> character bigrams
# n = 3 -> character trigrams

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
            self.n_gram_eu = np.zeros(shape=(self.corpus_size), dtype=float)
            self.n_gram_ca = np.zeros(shape=(self.corpus_size), dtype=float)
            self.n_gram_gl = np.zeros(shape=(self.corpus_size), dtype=float)
            self.n_gram_es = np.zeros(shape=(self.corpus_size), dtype=float)
            self.n_gram_en = np.zeros(shape=(self.corpus_size), dtype=float)
            self.n_gram_pt = np.zeros(shape=(self.corpus_size), dtype=float)

        elif self.n_gram_type == 2:
            self.n_gram_eu = np.zeros(shape=(self.corpus_size, self.corpus_size), dtype=float)
            self.n_gram_ca = np.zeros(shape=(self.corpus_size, self.corpus_size), dtype=float)
            self.n_gram_gl = np.zeros(shape=(self.corpus_size, self.corpus_size), dtype=float)
            self.n_gram_es = np.zeros(shape=(self.corpus_size, self.corpus_size), dtype=float)
            self.n_gram_en = np.zeros(shape=(self.corpus_size, self.corpus_size), dtype=float)
            self.n_gram_pt = np.zeros(shape=(self.corpus_size, self.corpus_size), dtype=float)

        elif self.n_gram_type == 3:
            self.n_gram_eu = np.zeros(shape=(self.corpus_size, self.corpus_size, self.corpus_size), dtype=float)
            self.n_gram_ca = np.zeros(shape=(self.corpus_size, self.corpus_size, self.corpus_size), dtype=float)
            self.n_gram_gl = np.zeros(shape=(self.corpus_size, self.corpus_size, self.corpus_size), dtype=float)
            self.n_gram_es = np.zeros(shape=(self.corpus_size, self.corpus_size, self.corpus_size), dtype=float)
            self.n_gram_en = np.zeros(shape=(self.corpus_size, self.corpus_size, self.corpus_size), dtype=float)
            self.n_gram_pt = np.zeros(shape=(self.corpus_size, self.corpus_size, self.corpus_size), dtype=float)


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
                        grams.append(char + data[i+1])

        elif self.n_gram_type == 3:
            for i, char in enumerate(data):
                if i < (len(data) - 2):
                    if (char in self.corpus) and (data[i+1] in self.corpus) and (data[i+2] in self.corpus):
                        grams.append(char + data[i+1] + data[i+2])

        return grams

    # Returns the index of the given char in the vocabulary
    def getCharIndex(self, char):
        try:
            return self.corpus.index(char)
        except ValueError as err:
            print(err)

    # Smooth-add the corresponding gram
    def smoothModel(self, language):
        if language == 'eu':
            self.n_gram_eu = self.n_gram_eu + self.smoothing
        elif language == 'ca':
            self.n_gram_ca = self.n_gram_ca + self.smoothing
        elif language == 'gl':
            self.n_gram_gl = self.n_gram_gl + self.smoothing
        elif language == 'es':
            self.n_gram_es = self.n_gram_es + self.smoothing
        elif language == 'en':
            self.n_gram_en = self.n_gram_en + self.smoothing
        elif language == 'pt':
            self.n_gram_pt = self.n_gram_pt + self.smoothing

    # increment the corresponding occurence in the gram
    def updateGram(self, position, language):
        if language == 'eu':
            if self.n_gram_type == 1:
                self.n_gram_eu[position[0]] += 1

            elif self.n_gram_type == 2:
                self.n_gram_eu[position[0]][position[1]] += 1

            elif self.n_gram_type == 3:
                self.n_gram_eu[position[0]][position[1]][position[2]] += 1

        elif language == 'ca':
            if self.n_gram_type == 1:
                self.n_gram_ca[position[0]] += 1

            elif self.n_gram_type == 2:
                self.n_gram_ca[position[0]][position[1]] += 1

            elif self.n_gram_type == 3:
                self.n_gram_ca[position[0]][position[1]][position[2]] += 1

        elif language == 'gl':
            if self.n_gram_type == 1:
                self.n_gram_gl[position[0]] += 1

            elif self.n_gram_type == 2:
                self.n_gram_gl[position[0]][position[1]] += 1

            elif self.n_gram_type == 3:
                self.n_gram_gl[position[0]][position[1]][position[2]] += 1

        elif language == 'es':
            if self.n_gram_type == 1:
                self.n_gram_es[position[0]] += 1

            elif self.n_gram_type == 2:
                self.n_gram_es[position[0]][position[1]] += 1

            elif self.n_gram_type == 3:
                self.n_gram_es[position[0]][position[1]][position[2]] += 1

        elif language == 'en':
            if self.n_gram_type == 1:
                self.n_gram_en[position[0]] += 1

            elif self.n_gram_type == 2:
                self.n_gram_en[position[0]][position[1]] += 1

            elif self.n_gram_type == 3:
                self.n_gram_en[position[0]][position[1]][position[2]] += 1

        elif language == 'pt':
            if self.n_gram_type == 1:
                self.n_gram_pt[position[0]] += 1

            elif self.n_gram_type == 2:
                self.n_gram_pt[position[0]][position[1]] += 1

            elif self.n_gram_type == 3:
                self.n_gram_pt[position[0]][position[1]][position[2]] += 1


    def train(self, grams, language):
        indexList = []

        print("Training EU: ")
        for gram in grams:
            for char in gram:
                indexList.append(self.getCharIndex(char))

            print("train() ", indexList)
            self.updateGram(indexList, language)

            indexList.clear()

    def run(self):
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
            print("Grams list: ", grams)
            print("Number of grams: ", len(grams))

            self.train(grams, language)


def main():
    print("Naive main.")

if __name__ == '__main__':
    main()
