class Output():

    def __init__(self, V, n, d):
        self.V = V
        self.n = n
        self.d = d

    # trace file as "trace_V_n_d.txt"
    def trace(self, tweetId, determinedClass, score, correctClass, label):
        with open('trace_' + str(self.V) + '_' + str(self.n) + '_' + str(self.d) + '.txt', 'a') as trace_file:
            trace_file.write(str(tweetId) + '  ' + determinedClass + '  ' + str(score) + '  ' + correctClass + ' ' + label + '\n')


    # overall evaluation as "eval_V_n_d.txt"
    # def overallEvaluation(self, ):

def main():
    print("Output main.")

if __name__ == '__main__':
    main()