from modules.neuralNet import NeuralNet
import time

def main():
    # Reading files
    train_file = input('Enter the training filename: ')
    test_file = input('Enter the test filename: ')

    # Setup
    model = NeuralNet(train_file, test_file)
    start_time = time.time()

    # Start Model
    model.cleanTrainData()
    model.cleanTestData()
    clean_time = time.time() - start_time
    print("--- Duration of Cleaning: %s seconds ---" % (clean_time))

    # Training
    model.train()
    train_time = time.time() - start_time - clean_time
    print("--- Duration of Training: %s seconds ---" % (train_time))

    # Testing
    model.runTest()
    test_time = time.time() - start_time - train_time
    print("--- Duration of Testing: %s seconds ---" % (test_time))

if __name__ == '__main__':
    main()