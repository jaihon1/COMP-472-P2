from modules.neuralNet import NeuralNet

def main():
    print("This is the main driver!")
    ## Reading files
    test_file = '/Users/dzhay/Github/COMP-472-P2/datasets/test/test-tweets-given.txt'
    train_file = '/Users/dzhay/Github/COMP-472-P2/datasets/train/training-tweets.txt'
    vocabulary = 0

    model = NeuralNet(vocabulary, train_file, test_file)

    model.cleanTrainData()
    model.cleanTestData()
    model.train()
    model.runTest()

if __name__ == '__main__':
    main()