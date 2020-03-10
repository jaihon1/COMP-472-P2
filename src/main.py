from modules.naiveBayes import NaiveBayes

def main():

    ## Reading files
    test_file = '/Users/dzhay/Github/COMP-472-P2/datasets/test/test-tweets-given.txt'
    train_file = '/Users/dzhay/Github/COMP-472-P2/datasets/train/training-tweets.txt'

    model = NaiveBayes(0, 1, 1, train_file, test_file)

    # print("Initial N gram EN: ")
    # print(model.getNGramEN())

    # print(model.getNGramEN())
    # print(model.getCorpus())
    model.runTrain()

    # model.smooth()


    # print("Initial N gram EN: ")
    # print(model.getNGramEN())

    # print("Initial N gram EU: ")
    # print(model.getNGramEU())

    model.runTest()

if __name__ == '__main__':
    main()
