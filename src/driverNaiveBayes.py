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
import time

def main():
    # Reading files
    train_file = input('Enter the training filename: ')
    test_file = input('Enter the test filename: ')

    print("--- Creating Model ---")
    model = NaiveBayes(2, 2, 0.3, train_file, test_file)

    # Initiate Timer
    start_time = time.time()

    # Training
    print("--- Started Training ---")
    model.runTrain()
    print("--- Ended Training ---")

    # Smoothing
    print("--- Started Smoothing ---")
    model.smooth()
    train_time = time.time() - start_time
    print("--- Ended Smoothing ---")
    print("--- Duration of Training: %s seconds ---" % (train_time))

    # Testing
    print("--- Started Testing ---")
    model.runTest()
    test_time = time.time() - start_time - train_time
    print("--- Ended Testing ---")
    print("--- Duration of Testing: %s seconds ---" % (test_time))


if __name__ == '__main__':
    main()
