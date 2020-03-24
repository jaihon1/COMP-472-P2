from modules.neuralNet import NeuralNet

def main():
    print("This is the main driver!")
    ## Reading files
    test_file = '/Users/dzhay/Github/COMP-472-P2/datasets/test/test-tweets-given.txt'
    train_file = '/Users/dzhay/Github/COMP-472-P2/datasets/train/training-tweets.txt'
    train_dataset = '/Users/dzhay/Github/COMP-472-P2/datasets/train/train-encoded-spaced.txt'
    test_dataset = '/Users/dzhay/Github/COMP-472-P2/datasets/test/test-encoded-spaced.txt'

    train_output = '/Users/dzhay/Github/COMP-472-P2/datasets/train/train-output.txt'
    test_ouput = '/Users/dzhay/Github/COMP-472-P2/datasets/test/test-output.txt'
    vocabulary = 0

    model = NeuralNet(vocabulary, train_file, test_file, train_dataset, test_dataset, train_output, test_ouput)


    # model.cleanTrainData()
    # model.runTest()
    # model.textToCsv()
    model.train()
    model.runTest()

if __name__ == '__main__':
    main()