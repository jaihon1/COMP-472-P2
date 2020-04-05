class Score():
    def __init__(self, language, score_value):
        self.language = language
        self.score_value = score_value

    def getScore(self):
        return self.score_value

    def getLanguage(self):
        return self.language


def main():
    print("Score main.")

if __name__ == '__main__':
    main()
