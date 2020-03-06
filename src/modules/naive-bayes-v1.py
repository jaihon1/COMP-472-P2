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


class NaiveBayes():
    def __init__(self, v, n, s, training_file_name, testing_file_name):
        self.vocabulary = v
        self.n_gram = n
        self.smoothing = s
        self.training_file_name = training_file_name
        self.testing_file_name = testing_file_name

    def getVocabulary(self):
        return self.vocabulary

    def getNGram(self):
        return self.n_gram

    def getTrainingFile(self):
        return self.training_file_name

    def getTestingFile(self):
        return self.testing_file_name

def main():
    print("Naive main.")

if __name__ == '__main__':
    main()
