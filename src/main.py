from modules.naiveBayes import NaiveBayes

def main():

    ## Reading files
    test_file = '../../datasets/test/test-tweets-given.txt'
    train_file = '/Users/dzhay/Github/COMP-472-P2/datasets/train/training-tweets.txt'

    model = NaiveBayes(0, 1, 1, train_file, test_file)

    # print(model.getNGramEN())
    # print(model.getCorpus())
    model.train()

if __name__ == '__main__':
    main()
