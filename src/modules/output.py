class Output():

    def __init__(self, V, n, d):
        self.V = V
        self.n = n
        self.d = d

    # trace file as "trace_V_n_d.txt"
    def trace(self, tweetId, determinedClass, score, correctClass, label):
        with open('trace_' + str(self.V) + '_' + str(self.n) + '_' + str(self.d) + '.txt', 'a') as trace_file:
            trace_file.write(str(tweetId) + '  ' + determinedClass + '  ' + '{:.2E}'.format(score) + '  ' + correctClass + ' ' + label + '\n')


    # overall evaluation as "eval_V_n_d.txt"
    def overallEvaluation(self, accuracy, classPrecision, classRecall, classF1, macroF1, weighedAvgF1):
        with open('eval_' + str(self.V) + '_' + str(self.n) + '_' + str(self.d) + '.txt', 'a') as eval_file:
            eval_file.write(str(round(accuracy, 4)) + '\n' + classPrecision + '\n' + classRecall + '\n' + classF1 + '\n' + str(round(macroF1, 4)) + '  ' + str(round(weighedAvgF1, 4)))

def main():
    print("Output main.")

if __name__ == '__main__':
    main()