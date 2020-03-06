from modules.naiveBayes import NaiveBayes

def main():

    ## Reading files
    test_file = '../../datasets/testing/test-tweets-given.txt'
    train_file = '../../datasets/testing/training-tweets.txt'

    model = NaiveBayes(2, 1, 1, test_file, train_file)

    print(model.getCorpus())



    # with open(test_file) as f:
    #     tweets = f.readlines()

    # for tweet in tweets:
    #     print(tweet)

if __name__ == '__main__':
    main()
