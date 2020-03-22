# Split tweets by word
# Clean words to only include vocabulary items
# Build dataset of those words and their respective label.
# Run the dataset into the NN as:
#   INPTUT as a word of max length ex. 20 where each word is encoded as following: (000 000 000 000 000 000 000 000 00) = (abc def ghi jkl mno pqr stu vwx yz)
#   OUTPUT (000000) = (eu, ca, gl, en, es, pt)
# For each tweet, splitted into valid words, guess which language it is. Average of all guesses of each word in that tweet.
# https://arxiv.org/abs/1903.07588
# https://medium.com/coinmonks/language-prediction-using-deep-neural-networks-42eb131444a5

import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import string
import statistics
import csv

from .wordEncoding import WordEncoding

class NeuralNet():
    def __init__(self, v, train_file_name, test_file_name, custom_train_file):
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.custom_train_file = custom_train_file
        self.corpus = None
        self.corpus_size = 0
        self.vocabulary_type = v

        self.generateVocabulary()


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


    # Convert word to lowercase
    def cleanWord_v1(self, word):
        if len(word) > 15:
            return None
        else:
            for char in word.lower():
                if (char not in self.corpus):
                    return None

        return word

    def cleanWord(self, word):
        if len(word) > 15:
            return None
        else:
            for char in word:
                if (char not in self.corpus):
                    return None

        return word


    def predictLanguage(self, word):
        # LANGUAGE ENCODING
        # eu = 0
        # ca = 1
        # gl = 2
        # es = 3
        # en = 4
        # pt = 5

        # Forward pass in NN
        return 1


    def train(self):
        print("Training")
        print('Reading train file')
        training_data = pd.read_csv(self.custom_train_file, header=None, delim_whitespace=True)
        training_data.columns = [*training_data.columns[:-1], 'label']

        input_data = training_data.drop(training_data.columns[len(training_data.columns) - 1], axis=1).values
        output_data = training_data[['label']].values

        print(input_data)
        print(output_data)

        # # Build Model
        # model = keras.models.Sequential()
        # model.add(keras.layers.Dense(50, input_dim=390, activation='relu'))
        # model.add(keras.layers.Dense(100, activation='relu'))
        # model.add(keras.layers.Dense(50, activation='relu'))
        # model.add(keras.layers.Dense(100, activation='relu'))
        # model.add(keras.layers.Dense(6, activation='linear'))

        # # Compile Model
        # model.compile(loss='mean_squared_error', optimizer='adam')

        # # Train Model
        # model.fit(input_data, output_data)

        # # error = model.evaluate(testing_data, expected_output)

        # # model.save('trained_model.h5')

        # # predictions = model.predict(new_data)



    def trainDriver(self):
        with open(self.custom_train_file) as f:
            tweets = f.readlines()

        for tweet in tweets:
            elements = tweet.split()

            if len(elements) > 0:
                # Get all info from a tweet
                word = elements[0]
                label = elements[1]

                self.train(word, label)

    def test(self, data):
        predictions = []

        for word in data:
            clean_result = self.cleanWord_v1(word)
            if clean_result is not None:
                language = self.predictLanguage(word)
                predictions.append(language)

        if not predictions:
            return 0
        else:
            return(statistics.mode(predictions))


    def runTest(self):
        with open(self.test_file_name) as f:
            tweets = f.readlines()

        for tweet in tweets:
            elements = tweet.split()

            if len(elements) > 0:
                # Get all info from a tweet
                userId = elements[0]
                username = elements[1]
                language = elements[2]
                data = ' '.join(elements[3:])
                data_split = data.split()

                guess = self.test(data_split)
                print(guess)


    def cleanTrainData(self):
        with open(self.train_file_name) as f:
            tweets = f.readlines()

        for tweet in tweets:
            elements = tweet.split()

            if len(elements) > 0:
                # Get all info from a tweet
                userId = elements[0]
                username = elements[1]
                language = elements[2]
                data = ' '.join(elements[3:])
                data_split = data.split()

                # Building train dataset with custom filter
                for word in data_split:
                    clean_result = self.cleanWord(word)
                    if clean_result is not None:
                        encoder = WordEncoding(word)
                        encoded_str = encoder.setup()

                        print(encoded_str)

                        with open('train.txt', 'a') as train_file:
                            train_file.write(encoded_str)
                            train_file.write(' ')
                            train_file.write(language)
                            train_file.write('\n')

        print("Done cleaning.")

    def textToCsv(self):
        with open(self.custom_train_file, 'r') as in_file:
            stripped = (line.strip() for line in in_file)
            lines = (line.split(",") for line in stripped if line)
            with open('train.csv', 'w') as out_file:
                writer = csv.writer(out_file)
                writer.writerows(lines)





def main():
    print("Neural Net main.")

if __name__ == '__main__':
    main()