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
    print("--- Started Cleaning ---")
    model.cleanTrainData()
    model.cleanTestData()
    clean_time = time.time() - start_time
    print("--- Ended Cleaning ---")
    print("--- Duration of Cleaning: %s seconds ---" % (clean_time))

    # Training
    print("--- Started Training ---")
    model.train()
    train_time = time.time() - start_time - clean_time
    print("--- Ended Training ---")
    print("--- Duration of Training: %s seconds ---" % (train_time))

    # Testing
    print("--- Started Testing ---")
    model.runTest()
    test_time = time.time() - start_time - train_time
    print("--- Ended Testing ---")
    print("--- Duration of Testing: %s seconds ---" % (test_time))

if __name__ == '__main__':
    main()