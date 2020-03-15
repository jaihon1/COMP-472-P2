from modules.neuralNet import NeuralNet

def main():
    print("This is the main driver!")
    ## Reading files
    test_file = '/Users/dzhay/Github/COMP-472-P2/datasets/test/test-tweets-given.txt'
    train_file = '/Users/dzhay/Github/COMP-472-P2/datasets/train/training-tweets.txt'
    custom_train_file = '/Users/dzhay/Github/COMP-472-P2/datasets/train/train-encoded.txt'
    vocabulary = 0

    model = NeuralNet(vocabulary, train_file, test_file, custom_train_file)

    model.cleanTrainData()

if __name__ == '__main__':
    main()