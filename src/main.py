from modules.naive-bayes-v1 import NaiveBayes

def main():

    ## Reading test file
    test_file = '../datasets/testing/test-tweets-given.txt'

    with open(test_file) as f:
        tweets = f.readlines()

    for tweet in tweets:
        print(tweet)

if __name__ == '__main__':
    main()
