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


### BEST CONFIGURATION (to date)
# (1, 3, 0.01) = 0.8455 accuracy

from modules.naiveBayes import NaiveBayes

def main():

    try:
        ## Reading files
        train_file = input('Enter the training filename: ')
        test_file = input('Enter the test filename: ')

        print(train_file)
        print(test_file)

        model = NaiveBayes(0, 3, 0.001, train_file, test_file)

        model.runTrain()

        model.smooth()

        model.runTest()

    except ValueError as err:
        print(err)

if __name__ == '__main__':
    main()
