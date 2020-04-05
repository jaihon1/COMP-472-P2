# Split tweets by word
# Clean words to only include vocabulary items
# Build dataset of those words and their respective label.
# Run the dataset into the NN as:
#   INPTUT as a word of max length ex. 20 where each word is encoded as following: (000 000 000 000 000 000 000 000 00) = (abc def ghi jkl mno pqr stu vwx yz)
#   OUTPUT (000000) = (eu, ca, gl, en, es, pt)
# For each tweet, splitted into valid words, guess which language it is. Average of all guesses of each word in that tweet.
# https://arxiv.org/abs/1903.07588
# https://medium.com/coinmonks/language-prediction-using-deep-neural-networks-42eb131444a5

# LANGUAGE ENCODING
# eu = 0
# ca = 1
# gl = 2
# es = 3
# en = 4
# pt = 5

import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import string
import statistics

from modules.outputMyModel import OutputMyModel
from .wordEncoding import WordEncoding
from .stats import Stats

# Toggles
TOGGLE_LOAD_MODEL = False

class NeuralNet():
    def __init__(self, train_file_name, test_file_name):
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.train_dataset_input = 'train-input-encoded.txt'
        self.train_dataset_output = 'train-output-encoded.txt'
        self.test_dataset_input = 'test-input-encoded.txt'
        self.test_dataset_output = 'test-output-encoded.txt'
        self.corpus = None
        self.corpus_size = 0
        self.vocabulary_type = 0

        self.generateVocabulary()

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


    # Convert word to lowercase and filter on corpus
    def cleanWord(self, word):
        if len(word) > 15:
            return None
        else:
            wordLower = word.lower()
            for char in wordLower:
                if (char not in self.corpus):
                    return None

            return wordLower

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
        else:
            return 'None'

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
        # Prepare datasets
        print('Preparing train dataset')
        training_data_input_raw = pd.read_csv(self.train_dataset_input, header=None, delim_whitespace=True)
        training_data_output_raw = pd.read_csv(self.train_dataset_output, header=None, delim_whitespace=True)

        train_input = training_data_input_raw.values
        train_output = training_data_output_raw.values
        print("DONE")

        print('Preparing test dataset')
        testing_data_input_raw = pd.read_csv(self.test_dataset_input, header=None, delim_whitespace=True)
        testing_data_output_raw = pd.read_csv(self.test_dataset_output, header=None, delim_whitespace=True)

        test_input = testing_data_input_raw.values
        test_output = testing_data_output_raw.values
        print("DONE")


        layer_1 = 256
        layer_2 = 256
        # layer_3 = 377
        # layer_4 = 16
        # layer_5 = 16

        # Build Model
        if TOGGLE_LOAD_MODEL:
            self.model = keras.models.load_model('not_bad_model_2.h5')

        else:
            self.model = keras.models.Sequential()
            self.model.add(keras.layers.Dense(layer_1, input_dim=390, activation='relu', name='layer_1'))
            self.model.add(keras.layers.Dense(layer_2, activation='relu', name='layer_2'))
            # self.model.add(keras.layers.Dense(layer_3, activation='relu', name='layer_3'))
            # self.model.add(keras.layers.Dense(layer_4, activation='relu', name='layer_4'))
            # self.model.add(keras.layers.Dense(layer_5, activation='relu', name='layer_5'))
            # self.model.add(keras.layers.Dense(200, activation='relu', name='layer_6'))
            # self.model.add(keras.layers.Dense(100, activation='relu', name='layer_7'))
            self.model.add(keras.layers.Dense(6, activation='softmax', name='output_layer'))

            # Compile Model
            self.model.compile(
                loss = 'categorical_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )

            # Create Logger
            RUN_NAME = 'run-with-{}-{}-nodes'.format(layer_1, layer_2)
            logger = keras.callbacks.TensorBoard(
                log_dir = 'logs/{}'.format(RUN_NAME),
                write_graph = True,
                histogram_freq = 5
            )

            # Create Checkpoint Saver
            checkpointer = keras.callbacks.ModelCheckpoint('{}.h5'.format(RUN_NAME), monitor='val_loss', verbose=1, save_best_only=True, mode='min')

            # Train Model
            self.model.fit(
                train_input,
                train_output,
                epochs=30,
                # validation_split=0.20,
                shuffle=True,
                # verbose=2,
                validation_data=(test_input, test_output),
                callbacks=[logger, checkpointer])

        # Evaluate Model
        error = self.model.evaluate(test_input, test_output, verbose=0)
        print('Test Error Rate: ', error)
        print(self.model.summary())

    def predictLanguage(self, word):
        encoder = WordEncoding(word)
        encoded_str = encoder.setup()

        input_vector = np.empty([1, 390], dtype=int)
        input_array = np.fromstring(encoded_str, dtype=int, sep=' ')
        input_vector[0] = input_array

        # Forward pass in NN
        predictions = self.model.predict(input_vector)
        for prediction in predictions:
            result = np.where(prediction == np.amax(prediction))

        return result[0][0]

    def test(self, data):
        predictions = []

        for word in data:
            clean_result = self.cleanWord(word)
            if clean_result is not None:
                language = self.predictLanguage(clean_result)
                predictions.append(language)

        return predictions

    def getHighestPrediction(self, predictions):
        try:
            prediction = statistics.mode(predictions)
            return prediction

        except ValueError:
            # print('Found 2 equally common values...')
            return None

    def getPredictionScore(self, prediction, predictions):
        score = 0

        if(prediction != None):
            # Get Prediction score
            total_score = 0
            size_predictions = len(predictions)

            for value in predictions:
                if value == prediction:
                    total_score += 1

            score = total_score / size_predictions

        return score


    def runTest(self):
        predictions = []
        targets = []

        i = 0

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

                i += 1
                if (i % 100) == 1:
                    print('Done tweets', i)

                predictions_model = self.test(data_split)
                if predictions_model:
                    prediction = self.getHighestPrediction(predictions_model)
                    scoreOfPrediction = self.getPredictionScore(prediction, predictions_model)

                else:
                    prediction = None
                    scoreOfPrediction = 0

                targets.append(language)
                predictions.append(self.languageToString(prediction))

                # Information for output files
                outputFile = OutputMyModel()
                predictedLanguage = self.languageToString(prediction)

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

                # Building train dataset custom
                for word in data_split:
                    clean_result = self.cleanWord(word)
                    if clean_result is not None:
                        encoder = WordEncoding(clean_result)
                        encoded_str = encoder.setup()
                        language_encoded = self.encodeLanguage(language)

                        # Create input files
                        with open(self.train_dataset_input, 'a') as train_file_input:
                            train_file_input.write(encoded_str)
                            train_file_input.write('\n')

                        # Create output file
                        with open(self.train_dataset_output, 'a') as train_file_output:
                            train_file_output.write(language_encoded)
                            train_file_output.write('\n')

        print("Done cleaning train dataset")

    def cleanTestData(self):
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

                # Building test dataset custom
                for word in data_split:
                    clean_result = self.cleanWord(word)
                    if clean_result is not None:
                        encoder = WordEncoding(clean_result)
                        encoded_str = encoder.setup()
                        language_encoded = self.encodeLanguage(language)

                        # Create input files
                        with open(self.test_dataset_input, 'a') as test_file_input:
                            test_file_input.write(encoded_str)
                            test_file_input.write('\n')

                        # Create output file
                        with open(self.test_dataset_output, 'a') as test_file_output:
                            test_file_output.write(language_encoded)
                            test_file_output.write('\n')

        print("Done cleaning test dataset")


def main():
    print("Neural Net main.")

if __name__ == '__main__':
    main()