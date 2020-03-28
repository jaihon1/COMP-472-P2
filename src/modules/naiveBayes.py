import string
import numpy as np
import math
from .score import Score

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

        self.eu_recall = 0
        self.ca_recall = 0
        self.gl_recall = 0
        self.es_recall = 0
        self.en_recall = 0
        self.pt_recall = 0

        self.eu_precision = 0
        self.ca_precision = 0
        self.gl_precision = 0
        self.es_precision = 0
        self.en_precision = 0
        self.pt_precision = 0

        self.eu_f1 = 0
        self.ca_f1 = 0
        self.gl_f1 = 0
        self.es_f1 = 0
        self.en_f1 = 0
        self.pt_f1 = 0

        self.eu_total_count = 0
        self.ca_total_count = 0
        self.gl_total_count = 0
        self.es_total_count = 0
        self.en_total_count = 0
        self.pt_total_count = 0

        self.macro_F1 = 0
        self.weighed_average_F1 = 0

        # Last column is for None -> no prediction
        # Rows are predictions
        # Columns are targets
        self.confusion_matrix = np.zeros((6, 6), dtype=int)



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

    def printAccuracy(self):
        print("GLOBAL ACCURACY", self.accuracy)

    def printPrecision(self):
        print('PRECISION')
        print("EU ", self.eu_precision)
        print("CA ", self.ca_precision)
        print("GL ", self.gl_precision)
        print("ES ", self.es_precision)
        print("EN ", self.en_precision)
        print("PT ", self.pt_precision)

    def printRecall(self):
        print('RECALL')
        print("EU ", self.eu_recall)
        print("CA ", self.ca_recall)
        print("GL ", self.gl_recall)
        print("ES ", self.es_recall)
        print("EN ", self.en_recall)
        print("PT ", self.pt_recall)

    def printF1(self):
        print('F1')
        print("EU ", self.eu_f1)
        print("CA ", self.ca_f1)
        print("GL ", self.gl_f1)
        print("ES ", self.es_f1)
        print("EN ", self.en_f1)
        print("PT ", self.pt_f1)

    def printMacroF1(self):
        print("MACRO F1", self.macro_F1)

    def printWeighedAverageF1(self):
        print("WEIGHED AVERAGE F1", self.weighed_average_F1)

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

    def buildConfusionMatrix(self, predictions, targets):
        for i, prediction in enumerate(predictions):
            if targets[i] == 'eu':
                if prediction == 'eu':
                    self.confusion_matrix[0][0] += 1
                elif prediction == 'ca':
                    self.confusion_matrix[1][0] += 1
                elif prediction == 'gl':
                    self.confusion_matrix[2][0] += 1
                elif prediction == 'es':
                    self.confusion_matrix[3][0] += 1
                elif prediction == 'en':
                    self.confusion_matrix[4][0] += 1
                elif prediction == 'pt':
                    self.confusion_matrix[5][0] += 1

            elif targets[i] == 'ca':
                if prediction == 'eu':
                    self.confusion_matrix[0][1] += 1
                elif prediction == 'ca':
                    self.confusion_matrix[1][1] += 1
                elif prediction == 'gl':
                    self.confusion_matrix[2][1] += 1
                elif prediction == 'es':
                    self.confusion_matrix[3][1] += 1
                elif prediction == 'en':
                    self.confusion_matrix[4][1] += 1
                elif prediction == 'pt':
                    self.confusion_matrix[5][1] += 1

            elif targets[i] == 'gl':
                if prediction == 'eu':
                    self.confusion_matrix[0][2] += 1
                elif prediction == 'ca':
                    self.confusion_matrix[1][2] += 1
                elif prediction == 'gl':
                    self.confusion_matrix[2][2] += 1
                elif prediction == 'es':
                    self.confusion_matrix[3][2] += 1
                elif prediction == 'en':
                    self.confusion_matrix[4][2] += 1
                elif prediction == 'pt':
                    self.confusion_matrix[5][2] += 1

            elif targets[i] == 'es':
                if prediction == 'eu':
                    self.confusion_matrix[0][3] += 1
                elif prediction == 'ca':
                    self.confusion_matrix[1][3] += 1
                elif prediction == 'gl':
                    self.confusion_matrix[2][3] += 1
                elif prediction == 'es':
                    self.confusion_matrix[3][3] += 1
                elif prediction == 'en':
                    self.confusion_matrix[4][3] += 1
                elif prediction == 'pt':
                    self.confusion_matrix[5][3] += 1

            elif targets[i] == 'en':
                if prediction == 'eu':
                    self.confusion_matrix[0][4] += 1
                elif prediction == 'ca':
                    self.confusion_matrix[1][4] += 1
                elif prediction == 'gl':
                    self.confusion_matrix[2][4] += 1
                elif prediction == 'es':
                    self.confusion_matrix[3][4] += 1
                elif prediction == 'en':
                    self.confusion_matrix[4][4] += 1
                elif prediction == 'pt':
                    self.confusion_matrix[5][4] += 1

            elif targets[i] == 'pt':
                if prediction == 'eu':
                    self.confusion_matrix[0][5] += 1
                elif prediction == 'ca':
                    self.confusion_matrix[1][5] += 1
                elif prediction == 'gl':
                    self.confusion_matrix[2][5] += 1
                elif prediction == 'es':
                    self.confusion_matrix[3][5] += 1
                elif prediction == 'en':
                    self.confusion_matrix[4][5] += 1
                elif prediction == 'pt':
                    self.confusion_matrix[5][5] += 1

    def calculateStats(self):
        # Total per category (Targets)
        column_sums = np.sum(self.confusion_matrix, axis = 0)
        # Total per category (Predictions)
        row_sums = np.sum(self.confusion_matrix, axis=1)
        table_sum = np.sum(self.confusion_matrix)
        diagonal_sum = np.trace(self.confusion_matrix, dtype=int)

        self.accuracy = diagonal_sum / table_sum

        self.eu_recall = self.confusion_matrix[0][0] / column_sums[0]
        self.ca_recall = self.confusion_matrix[1][1] / column_sums[1]
        self.gl_recall = self.confusion_matrix[2][2] / column_sums[2]
        self.es_recall = self.confusion_matrix[3][3] / column_sums[3]
        self.en_recall = self.confusion_matrix[4][4] / column_sums[4]
        self.pt_recall = self.confusion_matrix[5][5] / column_sums[5]

        self.eu_precision = self.confusion_matrix[0][0] / row_sums[0]
        self.ca_precision = self.confusion_matrix[1][1] / row_sums[1]
        self.gl_precision = self.confusion_matrix[2][2] / row_sums[2]
        self.es_precision = self.confusion_matrix[3][3] / row_sums[3]
        self.en_precision = self.confusion_matrix[4][4] / row_sums[4]
        self.pt_precision = self.confusion_matrix[5][5] / row_sums[5]

        self.eu_f1 = (2 * self.eu_precision * self.eu_recall) / (self.eu_precision + self.eu_recall)
        self.ca_f1 = (2 * self.ca_precision * self.ca_recall) / (self.ca_precision + self.ca_recall)
        self.gl_f1 = (2 * self.gl_precision * self.gl_recall) / (self.gl_precision + self.gl_recall)
        self.es_f1 = (2 * self.es_precision * self.es_recall) / (self.es_precision + self.es_recall)
        self.en_f1 = (2 * self.en_precision * self.en_recall) / (self.en_precision + self.en_recall)
        self.pt_f1 = (2 * self.pt_precision * self.pt_recall) / (self.pt_precision + self.pt_recall)

        self.macro_F1 = (self.eu_f1 + self.ca_f1 + self.gl_f1 + self.es_f1 + self.en_f1 + self.pt_f1) / 6
        self.weighed_average_F1 = (
            self.eu_f1*column_sums[0] +
            self.ca_f1*column_sums[1] +
            self.gl_f1*column_sums[2] +
            self.es_f1*column_sums[3] +
            self.en_f1*column_sums[4] +
            self.pt_f1*column_sums[5]) / 6





    def runTrain(self):
        # Train
        with open(self.train_file_name) as f:
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
        i = 0

        # Test
        with open(self.test_file_name) as f:
            tweets = f.readlines()

        for tweet in tweets:
            elementsTest = tweet.split()

            # Get all info from a tweet
            if len(elementsTest) > 0:
                userId = elementsTest[0]
                username = elementsTest[1]
                language = elementsTest[2]
                data = ' '.join(elementsTest[3:])

                grams = self.buildGram(data)
                # print("Tweet: ", grams, language)

                score_eu = Score('eu', self.test(grams, 'eu'))
                score_ca = Score('ca', self.test(grams, 'ca'))
                score_gl = Score('gl', self.test(grams, 'gl'))
                score_es = Score('es', self.test(grams, 'es'))
                score_en = Score('en', self.test(grams, 'en'))
                score_pt = Score('pt', self.test(grams, 'pt'))

                # print('score_eu', score_eu.score_value)
                # print('score_ca', score_ca.score_value)
                # print('score_gl', score_gl.score_value)
                # print('score_es', score_es.score_value)
                # print('score_en', score_en.score_value)
                # print('score_pt', score_pt.score_value)

                guesses = [score_eu, score_ca, score_gl, score_es, score_en, score_pt]
                guesses_score = [score_eu.score_value, score_ca.score_value, score_gl.score_value, score_es.score_value, score_en.score_value, score_pt.score_value]
                prediction = self.getMostAccurateLanguage(guesses_score)
                targets.append(language)
                predictions.append(guesses[prediction].language)

                # print('Original: ', language)
                # print('Model Guess: ', guesses[answer].language)
                # i += 1
                # if i == 20:
                #     break

        self.buildConfusionMatrix(predictions, targets)
        self.calculateStats()
        self.printAccuracy()
        self.printPrecision()
        self.printRecall()
        self.printF1()
        self.printMacroF1()
        self.printWeighedAverageF1()
        print(self.confusion_matrix)






def main():
    print("Naive main.")

if __name__ == '__main__':
    main()
