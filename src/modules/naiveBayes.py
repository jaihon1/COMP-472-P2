import string
import numpy as np
import math
from .score import Score
from modules.outputNaiveBayes import OutputNaiveBayes
from .stats import Stats

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
        self.grams_total = 0
        self.smoothing = s
        self.not_appear_char = 0 # single entry of characters not in the training set but isAlpha is true and vocab_type == 2
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name

        self.generateVocabulary()
        self.generateNgram()

        self.eu_total = 0
        self.ca_total = 0
        self.gl_total = 0
        self.es_total = 0
        self.en_total = 0
        self.pt_total = 0


    def getCorpus(self):
        return self.corpus

    def printNGrams(self):
        print("EU")
        print(self.n_gram_eu)
        print("CA")
        print(self.n_gram_ca)
        print("GL")
        print(self.n_gram_gl)
        print("ES")
        print(self.n_gram_es)
        print("EN")
        print(self.n_gram_en)
        print("PT")
        print(self.n_gram_pt)

    def printCounts(self):
        print("EU ", self.eu_total)
        print("CA ", self.ca_total)
        print("GL ", self.gl_total)
        print("ES ", self.es_total)
        print("EN ", self.en_total)
        print("PT ", self.pt_total)

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

        elif self.vocabulary_type == 2:
            self.corpus = self.getIsAlphaCorpus()
            self.corpus_size = len(self.corpus)

    def buildGramTest(self, data):
        grams = []
        if self.n_gram_type == 1:
            for char in data:
                if (char in self.corpus):
                    grams.append(char)
                elif char.isalpha():
                    self.not_appear_char += 1

        elif self.n_gram_type == 2:
            for i, char in enumerate(data):
                if i < (len(data) - 1):
                    if (char in self.corpus) and (data[i+1] in self.corpus):
                        grams.append(char + data[i+1])
                    elif char.isalpha() and (data[i+1].isalpha()):
                        self.not_appear_char += 1

        elif self.n_gram_type == 3:
            for i, char in enumerate(data):
                if i < (len(data) - 2):
                    if (char in self.corpus) and (data[i+1] in self.corpus) and (data[i+2] in self.corpus):
                        grams.append(char + data[i+1] + data[i+2])
                    elif char.isalpha() and (data[i+1].isalpha()) and (data[i+2].isalpha()):
                        self.not_appear_char += 1

        return grams

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

    # get the list of characters from the training set for which isalpha() == true
    def getIsAlphaCorpus(self):
        corpus = list()
        # Train
        with open(self.train_file_name, encoding='utf8') as f:
            tweets = f.readlines()

        for tweet in tweets:
            elementsTrain = tweet.split()


            if len(elementsTrain) > 0:
                # Get all info from a tweet
                data = ' '.join(elementsTrain[3:])
                for char in data:
                    if char.isalpha() and char not in corpus:
                        corpus.append(char)
        return corpus

    # Smooth-add the corresponding gram
    def smooth(self):
        self.n_gram_eu = self.n_gram_eu + self.smoothing
        self.n_gram_ca = self.n_gram_ca + self.smoothing
        self.n_gram_gl = self.n_gram_gl + self.smoothing
        self.n_gram_es = self.n_gram_es + self.smoothing
        self.n_gram_en = self.n_gram_en + self.smoothing
        self.n_gram_pt = self.n_gram_pt + self.smoothing

        self.eu_total +=  self.corpus_size * self.smoothing
        self.ca_total +=  self.corpus_size * self.smoothing
        self.gl_total +=  self.corpus_size * self.smoothing
        self.es_total +=  self.corpus_size * self.smoothing
        self.en_total +=  self.corpus_size * self.smoothing
        self.pt_total +=  self.corpus_size * self.smoothing


    def getScore(self, position, language):
        score = 0

        if language == 'eu':
            if self.n_gram_type == 1:
                score = self.n_gram_eu[position[0]] / self.eu_total

            elif self.n_gram_type == 2:
                score = self.n_gram_eu[position[0]][position[1]] / self.eu_total

            elif self.n_gram_type == 3:
                score = self.n_gram_eu[position[0]][position[1]][position[2]] / self.eu_total

        if language == 'ca':
            if self.n_gram_type == 1:
                score = self.n_gram_ca[position[0]] / self.ca_total

            elif self.n_gram_type == 2:
                score = self.n_gram_ca[position[0]][position[1]] / self.ca_total

            elif self.n_gram_type == 3:
                score = self.n_gram_ca[position[0]][position[1]][position[2]] / self.ca_total

        if language == 'gl':
            if self.n_gram_type == 1:
                score = self.n_gram_gl[position[0]] / self.gl_total

            elif self.n_gram_type == 2:
                score = self.n_gram_gl[position[0]][position[1]] / self.gl_total

            elif self.n_gram_type == 3:
                score = self.n_gram_gl[position[0]][position[1]][position[2]] / self.gl_total

        if language == 'es':
            if self.n_gram_type == 1:
                score = self.n_gram_es[position[0]] / self.es_total

            elif self.n_gram_type == 2:
                score = self.n_gram_es[position[0]][position[1]] / self.es_total

            elif self.n_gram_type == 3:
                score = self.n_gram_es[position[0]][position[1]][position[2]] / self.es_total

        if language == 'en':
            if self.n_gram_type == 1:
                score = self.n_gram_en[position[0]] / self.en_total

            elif self.n_gram_type == 2:
                score = self.n_gram_en[position[0]][position[1]] / self.en_total

            elif self.n_gram_type == 3:
                score = self.n_gram_en[position[0]][position[1]][position[2]] / self.en_total

        if language == 'pt':
            if self.n_gram_type == 1:
                score = self.n_gram_pt[position[0]] / self.pt_total

            elif self.n_gram_type == 2:
                score = self.n_gram_pt[position[0]][position[1]] / self.pt_total

            elif self.n_gram_type == 3:
                score = self.n_gram_pt[position[0]][position[1]][position[2]] / self.pt_total

        return score

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


    def updateCount(self, language):
        if language == 'eu':
            self.eu_total += 1

        elif language == 'ca':
            self.ca_total += 1

        elif language == 'gl':
            self.gl_total += 1

        elif language == 'es':
            self.es_total += 1

        elif language == 'en':
            self.en_total += 1

        elif language == 'pt':
            self.pt_total += 1

    def train(self, grams, language):
        indexList = []

        for gram in grams:
            for char in gram:
                indexList.append(self.getCharIndex(char))

            # print("train() ", indexList)
            self.updateCount(language)
            self.updateGram(indexList, language)

            indexList.clear()

    def test(self, grams, language):
        indexList = []
        totalScore = 0

        for gram in grams:
            for char in gram:
                indexList.append(self.getCharIndex(char))

            totalScore += math.log10(self.getScore(indexList, language))
            indexList.clear()

        return totalScore

    def getMostAccurateLanguage(self, guesses):
        index = np.argmax(guesses)
        return index

    def runTrain(self):
        # Train
        with open(self.train_file_name, encoding='utf8') as f:
            tweets = f.readlines()

        for tweet in tweets:
            elementsTrain = tweet.split()

            if len(elementsTrain) > 0:
                # Get all info from a tweet
                userId = elementsTrain[0]
                username = elementsTrain[1]
                language = elementsTrain[2]
                data = ' '.join(elementsTrain[3:])

                grams = self.buildGram(data)

                self.train(grams, language)


    def runTest(self):
        predictions = []
        targets = []

        # Test
        with open(self.test_file_name, encoding='utf8') as f:
            tweets = f.readlines()

        for tweet in tweets:
            elementsTest = tweet.split()

            # Get all info from a tweet
            if len(elementsTest) > 0:
                userId = elementsTest[0]
                username = elementsTest[1]
                language = elementsTest[2]
                data = ' '.join(elementsTest[3:])

                if (self.vocabulary_type == 2):
                    grams = self.buildGramTest(data)
                else:
                    grams = self.buildGram(data)

                score_eu = Score('eu', self.test(grams, 'eu'))
                score_ca = Score('ca', self.test(grams, 'ca'))
                score_gl = Score('gl', self.test(grams, 'gl'))
                score_es = Score('es', self.test(grams, 'es'))
                score_en = Score('en', self.test(grams, 'en'))
                score_pt = Score('pt', self.test(grams, 'pt'))

                guesses = [score_eu, score_ca, score_gl, score_es, score_en, score_pt]
                guesses_score = [score_eu.score_value, score_ca.score_value, score_gl.score_value, score_es.score_value, score_en.score_value, score_pt.score_value]
                prediction = self.getMostAccurateLanguage(guesses_score)
                targets.append(language)
                predictions.append(guesses[prediction].language)

                # Information for output files
                outputFile = OutputNaiveBayes(self.vocabulary_type, self.n_gram_type, self.smoothing)
                predictedLanguage = guesses[prediction].language

                # score of predictedLanguage
                for score in guesses:
                    if (predictedLanguage == score.getLanguage()):
                        scoreOfPrediction = score.getScore()

                # correct/wrong label
                if(predictedLanguage == language):
                    label = 'correct'
                else:
                    label = 'wrong'

                # trace file
                outputFile.trace(userId, predictedLanguage, scoreOfPrediction, language, label)

        stats = Stats(predictions, targets)
        stats.buildConfusionMatrix()
        stats.calculateStats()
        stats.printStats()

        # eval file
        outputFile.overallEvaluation(stats.accuracy, stats.outputClassPrecisions(), stats.outputClassRecalls(), stats.outputClassF1(), stats.macro_F1, stats.weighed_average_F1)



def main():
    print("Naive main.")

if __name__ == '__main__':
    main()
