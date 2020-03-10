from modules.naiveBayes import NaiveBayes

def main():

    ## Reading files
    test_file = '/Users/dzhay/Github/COMP-472-P2/datasets/test/test-tweets-given.txt'
    train_file = '/Users/dzhay/Github/COMP-472-P2/datasets/train/training-tweets.txt'

    model = NaiveBayes(1, 2, 1, train_file, test_file)

    model.runTrain()

    model.runTest()

if __name__ == '__main__':
    main()
