from modules.neuralNet import NeuralNet
import time

def main():
    ## Reading files
    test_file = '/Users/dzhay/Github/COMP-472-P2/datasets/test/test-tweets-given.txt'
    train_file = '/Users/dzhay/Github/COMP-472-P2/datasets/train/training-tweets.txt'
    vocabulary = 0

    # Setup
    model = NeuralNet(vocabulary, train_file, test_file)
    start_time = time.time()

    # Start Model
    model.cleanTrainData()
    model.cleanTestData()
    clean_time = time.time() - start_time
    print("--- Duration of Cleaning: %s seconds ---" % (clean_time))

    model.train()
    train_time = time.time() - start_time - clean_time
    print("--- Duration of Training: %s seconds ---" % (train_time))

    model.runTest()
    test_time = time.time() - start_time - train_time
    print("--- Duration of Testing: %s seconds ---" % (test_time))

if __name__ == '__main__':
    main()