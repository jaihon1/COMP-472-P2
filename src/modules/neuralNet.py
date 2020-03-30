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
from .stats import Stats


TOGGLE_LOAD_MODEL = True
TOGGLE_LOAD_AND_TRAIN_MODEL = False

class NeuralNet():
    def __init__(self, v, train_file_name, test_file_name, train_dataset, test_dataset, train_output, test_output):
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.corpus = None
        self.corpus_size = 0
        self.vocabulary_type = v

        self.generateVocabulary()

        self.train_dataset_output = train_output
        self.test_dataset_output = test_output

        self.model = None

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
            wordLower = word.lower()
            for char in wordLower:
                if (char not in self.corpus):
                    return None

            return wordLower

    def cleanWord(self, word):
        if len(word) > 15:
            return None
        else:
            for char in word:
                if (char not in self.corpus):
                    return None

        return word

    def languageToString(self, languageNum):
        if languageNum == 0:
            return 'eu'
        if languageNum == 1:
            return 'ca'
        if languageNum == 2:
            return 'gl'
        if languageNum == 3:
            return 'es'
        if languageNum == 4:
            return 'en'
        if languageNum == 5:
            return 'pt'

    def stringToLanguage(self, languageString):
        if languageString == 'eu':
            return '0'
        if languageString == 'ca':
            return '1'
        if languageString == 'gl':
            return '2'
        if languageString == 'es':
            return '3'
        if languageString == 'en':
            return '4'
        if languageString == 'pt':
            return '5'


    def encodeLanguage(self, language):
        if language == 'eu':
            return '1 0 0 0 0 0'

        elif language == 'ca':
            return '0 1 0 0 0 0'

        elif language == 'gl':
            return '0 0 1 0 0 0'

        elif language == 'es':
            return '0 0 0 1 0 0'

        elif language == 'en':
            return '0 0 0 0 1 0'

        elif language == 'pt':
            return '0 0 0 0 0 1'


    def train(self):
        print("Training")
        print('Preparing train dataset')
        training_data = pd.read_csv(self.train_dataset, header=None, delim_whitespace=True)
        training_data_out = pd.read_csv(self.train_dataset_output, header=None, delim_whitespace=True)
        # training_data.columns = [*training_data.columns[:-1], 'label']

        # train_input_data = training_data.drop(training_data.columns[len(training_data.columns) - 1], axis=1).values
        train_input_data = training_data.values
        train_output_data = training_data_out.values
        # train_output_data = training_data[['label']].values


        # print(train_input_data)
        # print(len(train_input_data))
        # print(train_output_data)
        # print(len(train_output_data))
        print("DONE")

        print('Preparing test dataset')
        testing_data = pd.read_csv(self.test_dataset, header=None, delim_whitespace=True)
        testing_data_out = pd.read_csv(self.test_dataset_output, header=None, delim_whitespace=True)
        # testing_data.columns = [*testing_data.columns[:-1], 'label']

        # test_input_data = testing_data.drop(testing_data.columns[len(testing_data.columns) - 1], axis=1).values
        test_input_data = testing_data.values
        test_output_data = testing_data_out.values
        # test_output_data = testing_data[['label']].values

        # print(test_input_data)
        # print(len(train_input_data))
        # print(test_output_data)
        # print(len(test_output_data))
        print("DONE")

        # Build Model

        if TOGGLE_LOAD_MODEL:
            self.model = keras.models.load_model('trained_model.h5')

        elif TOGGLE_LOAD_AND_TRAIN_MODEL:
            self.model = keras.models.load_model('trained_model.h5')

            # Create Logger
            RUN_NAME = 'run 1 with 50 nodes'
            logger = keras.callbacks.TensorBoard(
                log_dir = 'logs/{}'.format(RUN_NAME),
                write_graph = True,
                histogram_freq = 5
            )

            # Train Model
            self.model.fit(
                train_input_data,
                train_output_data,
                epochs=15,
                shuffle=True,
                verbose=2,
                validation_data=(test_input_data, test_output_data),
                callbacks=[logger])

            # Save model
            self.model.save('trained_model.h5')
            print('Saved model')

        else:
            self.model = keras.models.Sequential()
            self.model.add(keras.layers.Dense(500, input_dim=390, activation='relu', name='layer1_1'))
            self.model.add(keras.layers.Dense(200, activation='relu', name='layer1_2'))
            self.model.add(keras.layers.Dense(500, activation='relu', name='layer1_3'))
            # self.model.add(keras.layers.Dense(100, activation='relu', name='layer1_4'))
            # self.model.add(keras.layers.Dense(400, activation='relu', name='layer1_5'))
            # self.model.add(keras.layers.Dense(200, activation='relu', name='layer1_6'))
            # self.model.add(keras.layers.Dense(100, activation='relu', name='layer1_7'))
            self.model.add(keras.layers.Dense(6, activation='softmax', name='output_layer'))

            # Compile Model
            self.model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )

            # Create Logger
            RUN_NAME = 'run 1 with 600 400 600 nodes, adam'
            logger = keras.callbacks.TensorBoard(
                log_dir = 'logs/{}'.format(RUN_NAME),
                write_graph = True,
                histogram_freq = 5
            )

            # Create Checkpointer
            checkpointer = keras.callbacks.ModelCheckpoint('trained_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

            # Train Model
            self.model.fit(
                train_input_data,
                train_output_data,
                epochs=10,
                # validation_split=0.20,
                shuffle=True,
                # verbose=2,
                validation_data=(test_input_data, test_output_data),
                callbacks=[logger, checkpointer])

            # Save model
            # self.model.save('trained_model.h5')
            print('Saved model')

            print('hello', self.languageToString(self.predictLanguage('hello')))
            print('hola', self.languageToString(self.predictLanguage('hola')))
            print('bonjour', self.languageToString(self.predictLanguage('bonjour')))
            print('bueno', self.languageToString(self.predictLanguage('bueno')))


        # Evaluate Model
        error = self.model.evaluate(test_input_data, test_output_data, verbose=0 )
        print('Test Error Rate: ', error)


    def trainDriver(self):
        with open(self.train_dataset) as f:
            tweets = f.readlines()

        for tweet in tweets:
            elements = tweet.split()

            if len(elements) > 0:
                # Get all info from a tweet
                word = elements[0]
                label = elements[1]

                self.train()

    def predictLanguage(self, word):
        # LANGUAGE ENCODING
        # eu = 0
        # ca = 1
        # gl = 2
        # es = 3
        # en = 4
        # pt = 5

        encoder = WordEncoding(word)
        encoded_str = encoder.setup()

        test_input = np.empty([1, 390], dtype=int)
        test_array = np.fromstring(encoded_str, dtype=int, sep=' ')
        test_input[0] = test_array

        # Forward pass in NN
        predictions = self.model.predict(test_input)
        for prediction in predictions:
            result = np.where(prediction == np.amax(prediction))

        # print('Predictions', predictions)
        return result[0][0]


    def test(self, data):
        predictions = []

        for word in data:
            clean_result = self.cleanWord_v1(word)
            if clean_result is not None:
                language = self.predictLanguage(clean_result)
                predictions.append(language)

        if not predictions:
            return 0
        else:
            try:
                guess = statistics.mode(predictions)
                return guess

            except ValueError:
                # print('Found 2 equally common values...')
                return None


    def runTest(self):
        predictions = []
        targets = []
        i = 0

        countEU = 0
        countCA = 0
        countGL = 0
        countES = 0
        countEN = 0
        countPT = 0

        write = True
        TWEET_LIMIT = 1000

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

                if language == 'eu':
                    if countEU < TWEET_LIMIT:
                        write = True
                        countEU += 1

                elif language == 'ca':
                    if countCA < TWEET_LIMIT:
                        write = True
                        countCA += 1

                elif language == 'gl':
                    if countGL < TWEET_LIMIT:
                        write = True
                        countGL += 1

                elif language == 'es':
                    if countES < TWEET_LIMIT:
                        write = True
                        countES += 1

                elif language == 'en':
                    if countEN < TWEET_LIMIT:
                        write = True
                        countEN += 1

                elif language == 'pt':
                    if countPT < TWEET_LIMIT:
                        write = True
                        countPT += 1

                if (i % 100) == 1:
                    print('Done tweets', i)

                if write:
                    i += 1
                    if (i % 100) == 1:
                        print('Done tweets', i)

                    prediction = self.test(data_split)
                    targets.append(language)
                    predictions.append(self.languageToString(prediction))

                    # write = False

        stats = Stats(predictions, targets)
        stats.buildConfusionMatrix()
        stats.calculateStats()
        stats.printStats()



    def cleanData(self):
        countEU = 0
        countCA = 0
        countGL = 0
        countES = 0
        countEN = 0
        countPT = 0

        write = True
        WORD_LIMIT = 12000

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

                # language = 'pt'
                # data = ' '.join(elements[0:])
                # data_split = data.split()

                # if language == 'eu':
                #     self.countEU += 1

                # elif language == 'ca':
                #     self.countCA += 1

                # elif language == 'gl':
                #     self.countGL += 1

                # elif language == 'es':
                #     self.countES += 1

                # elif language == 'en':
                #     self.countEN += 1

                # elif language == 'pt':
                #     self.countPT += 1

                # Building train dataset with custom filter
                for word in data_split:
                    clean_result = self.cleanWord(word)
                    if clean_result is not None:
                        encoder = WordEncoding(word)
                        encoded_str = encoder.setup()

                        language_encoded = self.encodeLanguage(language)

                        if language == 'eu':
                            if countEU < WORD_LIMIT:
                                write = True
                                countEU += 1

                        elif language == 'ca':
                            if countCA < WORD_LIMIT:
                                write = True
                                countCA += 1

                        elif language == 'gl':
                            if countGL < WORD_LIMIT:
                                write = True
                                countGL += 1

                        elif language == 'es':
                            if countES < WORD_LIMIT:
                                write = True
                                countES += 1

                        elif language == 'en':
                            if countEN < WORD_LIMIT:
                                write = True
                                countEN += 1

                        elif language == 'pt':
                            if countPT < WORD_LIMIT:
                                write = True
                                countPT += 1

                        if write == True:
                            # with open('train-encoded-spaced-full.txt', 'a') as train_file:
                            with open('train-output-full-code.txt', 'a') as train_file:
                                # train_file.write(encoded_str)
                                # train_file.write(' ')
                                train_file.write(language_encoded)
                                train_file.write('\n')

                            # write = False


        print("Done cleaning.")

    def textToCsv(self):
        with open(self.train_dataset, 'r') as in_file:
            stripped = (line.strip() for line in in_file)
            lines = (line.split(",") for line in stripped if line)
            with open('train.csv', 'w') as out_file:
                writer = csv.writer(out_file)
                writer.writerows(lines)




def main():
    print("Neural Net main.")

if __name__ == '__main__':
    main()