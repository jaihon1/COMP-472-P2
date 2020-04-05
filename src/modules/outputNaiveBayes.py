class OutputNaiveBayes():

    def __init__(self, V, n, d):
        self.V = V
        self.n = n
        self.d = d

    # trace file as "trace_V_n_d.txt"
    def trace(self, tweetId, determinedClass, score, correctClass, label):
        with open('trace_' + str(self.V) + '_' + str(self.n) + '_' + str(self.d) + '.txt', 'a') as trace_file:
            trace_file.write(str(tweetId) + '  ' + str(determinedClass) + '  ' + '{:.2E}'.format(score) + '  ' + correctClass + ' ' + label + '\n')


    # overall evaluation as "eval_V_n_d.txt"
    def overallEvaluation(self, accuracy, classPrecision, classRecall, classF1, macroF1, weighedAvgF1):
        with open('eval_' + str(self.V) + '_' + str(self.n) + '_' + str(self.d) + '.txt', 'a') as eval_file:
            eval_file.write(str('{:.4f}'.format(accuracy)) + '\n' + classPrecision + '\n' + classRecall + '\n' + classF1 + '\n' + str('{:.4f}'.format(macroF1)) + '  ' + str('{:.4f}'.format(weighedAvgF1)))

def main():
    print("Output main.")

if __name__ == '__main__':
    main()