import numpy as np

class Stats():
    def __init__(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets

        self.eu_recall = 0
        self.ca_recall = 0
        self.gl_recall = 0
        self.es_recall = 0
        self.en_recall = 0
        self.pt_recall = 0

        self.eu_precision = 0
        self.ca_precision = 0
        self.gl_precision = 0
        self.es_precision = 0
        self.en_precision = 0
        self.pt_precision = 0

        self.eu_f1 = 0
        self.ca_f1 = 0
        self.gl_f1 = 0
        self.es_f1 = 0
        self.en_f1 = 0
        self.pt_f1 = 0

        self.eu_total_count = 0
        self.ca_total_count = 0
        self.gl_total_count = 0
        self.es_total_count = 0
        self.en_total_count = 0
        self.pt_total_count = 0

        self.macro_F1 = 0
        self.weighed_average_F1 = 0

        self.accuracy = 0

        # Last column is for None -> no prediction
        # Rows are predictions
        # Columns are targets
        self.confusion_matrix = np.zeros((7, 6), dtype=int)


    def printAccuracy(self):
        print("GLOBAL ACCURACY", self.accuracy)

    def printPrecision(self):
        print('PRECISION')
        print("EU ", self.eu_precision)
        print("CA ", self.ca_precision)
        print("GL ", self.gl_precision)
        print("ES ", self.es_precision)
        print("EN ", self.en_precision)
        print("PT ", self.pt_precision)

    def printRecall(self):
        print('RECALL')
        print("EU ", self.eu_recall)
        print("CA ", self.ca_recall)
        print("GL ", self.gl_recall)
        print("ES ", self.es_recall)
        print("EN ", self.en_recall)
        print("PT ", self.pt_recall)

    def printF1(self):
        print('F1')
        print("EU ", self.eu_f1)
        print("CA ", self.ca_f1)
        print("GL ", self.gl_f1)
        print("ES ", self.es_f1)
        print("EN ", self.en_f1)
        print("PT ", self.pt_f1)

    def printMacroF1(self):
        print("MACRO F1", self.macro_F1)

    def printWeighedAverageF1(self):
        print("WEIGHED AVERAGE F1", self.weighed_average_F1)

    def printConfusionMatrix(self):
        print("CONFUSION MATRIX")
        print(self.confusion_matrix)

    def printStats(self):
        self.printAccuracy()
        self.printPrecision()
        self.printRecall()
        self.printF1()
        self.printMacroF1()
        self.printWeighedAverageF1()
        self.printConfusionMatrix()

    def buildConfusionMatrix(self):
        for i, prediction in enumerate(self.predictions):
            if self.targets[i] == 'eu':
                if prediction == 'eu':
                    self.confusion_matrix[0][0] += 1
                elif prediction == 'ca':
                    self.confusion_matrix[1][0] += 1
                elif prediction == 'gl':
                    self.confusion_matrix[2][0] += 1
                elif prediction == 'es':
                    self.confusion_matrix[3][0] += 1
                elif prediction == 'en':
                    self.confusion_matrix[4][0] += 1
                elif prediction == 'pt':
                    self.confusion_matrix[5][0] += 1
                else:
                    self.confusion_matrix[6][0] += 1

            elif self.targets[i] == 'ca':
                if prediction == 'eu':
                    self.confusion_matrix[0][1] += 1
                elif prediction == 'ca':
                    self.confusion_matrix[1][1] += 1
                elif prediction == 'gl':
                    self.confusion_matrix[2][1] += 1
                elif prediction == 'es':
                    self.confusion_matrix[3][1] += 1
                elif prediction == 'en':
                    self.confusion_matrix[4][1] += 1
                elif prediction == 'pt':
                    self.confusion_matrix[5][1] += 1
                else:
                    self.confusion_matrix[6][1] += 1

            elif self.targets[i] == 'gl':
                if prediction == 'eu':
                    self.confusion_matrix[0][2] += 1
                elif prediction == 'ca':
                    self.confusion_matrix[1][2] += 1
                elif prediction == 'gl':
                    self.confusion_matrix[2][2] += 1
                elif prediction == 'es':
                    self.confusion_matrix[3][2] += 1
                elif prediction == 'en':
                    self.confusion_matrix[4][2] += 1
                elif prediction == 'pt':
                    self.confusion_matrix[5][2] += 1
                else:
                    self.confusion_matrix[6][2] += 1

            elif self.targets[i] == 'es':
                if prediction == 'eu':
                    self.confusion_matrix[0][3] += 1
                elif prediction == 'ca':
                    self.confusion_matrix[1][3] += 1
                elif prediction == 'gl':
                    self.confusion_matrix[2][3] += 1
                elif prediction == 'es':
                    self.confusion_matrix[3][3] += 1
                elif prediction == 'en':
                    self.confusion_matrix[4][3] += 1
                elif prediction == 'pt':
                    self.confusion_matrix[5][3] += 1
                else:
                    self.confusion_matrix[6][3] += 1

            elif self.targets[i] == 'en':
                if prediction == 'eu':
                    self.confusion_matrix[0][4] += 1
                elif prediction == 'ca':
                    self.confusion_matrix[1][4] += 1
                elif prediction == 'gl':
                    self.confusion_matrix[2][4] += 1
                elif prediction == 'es':
                    self.confusion_matrix[3][4] += 1
                elif prediction == 'en':
                    self.confusion_matrix[4][4] += 1
                elif prediction == 'pt':
                    self.confusion_matrix[5][4] += 1
                else:
                    self.confusion_matrix[6][4] += 1

            elif self.targets[i] == 'pt':
                if prediction == 'eu':
                    self.confusion_matrix[0][5] += 1
                elif prediction == 'ca':
                    self.confusion_matrix[1][5] += 1
                elif prediction == 'gl':
                    self.confusion_matrix[2][5] += 1
                elif prediction == 'es':
                    self.confusion_matrix[3][5] += 1
                elif prediction == 'en':
                    self.confusion_matrix[4][5] += 1
                elif prediction == 'pt':
                    self.confusion_matrix[5][5] += 1
                else:
                    self.confusion_matrix[6][5] += 1

    def calculateStats(self):
        # Total per category (Targets)
        column_sums = np.sum(self.confusion_matrix, axis = 0)
        # Total per category (Predictions)
        row_sums = np.sum(self.confusion_matrix, axis=1)
        table_sum = np.sum(self.confusion_matrix)
        diagonal_sum = np.trace(self.confusion_matrix, dtype=int)

        self.accuracy = diagonal_sum / table_sum

        self.eu_recall = self.confusion_matrix[0][0] / column_sums[0] if column_sums[0] else 0
        self.ca_recall = self.confusion_matrix[1][1] / column_sums[1] if column_sums[1] else 0
        self.gl_recall = self.confusion_matrix[2][2] / column_sums[2] if column_sums[2] else 0
        self.es_recall = self.confusion_matrix[3][3] / column_sums[3] if column_sums[3] else 0
        self.en_recall = self.confusion_matrix[4][4] / column_sums[4] if column_sums[4] else 0
        self.pt_recall = self.confusion_matrix[5][5] / column_sums[5] if column_sums[5] else 0

        self.eu_precision = self.confusion_matrix[0][0] / row_sums[0] if row_sums[0] else 0
        self.ca_precision = self.confusion_matrix[1][1] / row_sums[1] if row_sums[1] else 0
        self.gl_precision = self.confusion_matrix[2][2] / row_sums[2] if row_sums[2] else 0
        self.es_precision = self.confusion_matrix[3][3] / row_sums[3] if row_sums[3] else 0
        self.en_precision = self.confusion_matrix[4][4] / row_sums[4] if row_sums[4] else 0
        self.pt_precision = self.confusion_matrix[5][5] / row_sums[5] if row_sums[5] else 0

        self.eu_f1 = (2 * self.eu_precision * self.eu_recall) / (self.eu_precision + self.eu_recall) if (self.eu_precision + self.eu_recall) else 0
        self.ca_f1 = (2 * self.ca_precision * self.ca_recall) / (self.ca_precision + self.ca_recall) if (self.ca_precision + self.ca_recall) else 0
        self.gl_f1 = (2 * self.gl_precision * self.gl_recall) / (self.gl_precision + self.gl_recall) if (self.gl_precision + self.gl_recall) else 0
        self.es_f1 = (2 * self.es_precision * self.es_recall) / (self.es_precision + self.es_recall) if (self.es_precision + self.es_recall) else 0
        self.en_f1 = (2 * self.en_precision * self.en_recall) / (self.en_precision + self.en_recall) if (self.en_precision + self.en_recall) else 0
        self.pt_f1 = (2 * self.pt_precision * self.pt_recall) / (self.pt_precision + self.pt_recall) if (self.pt_precision + self.pt_recall) else 0

        self.macro_F1 = (self.eu_f1 + self.ca_f1 + self.gl_f1 + self.es_f1 + self.en_f1 + self.pt_f1) / 6
        self.weighed_average_F1 = (
            self.eu_f1*column_sums[0] +
            self.ca_f1*column_sums[1] +
            self.gl_f1*column_sums[2] +
            self.es_f1*column_sums[3] +
            self.en_f1*column_sums[4] +
            self.pt_f1*column_sums[5]) / table_sum

    def outputClassPrecisions(self):
        # eu-P, ca-P, gl-P, es-P, en-P and pt-P
        classPrecision = str('{:.4f}'.format(self.eu_precision)) + '  ' + str('{:.4f}'.format(self.ca_precision)) + '  ' + str('{:.4f}'.format(self.gl_precision)) + '  ' 
        classPrecision += str('{:.4f}'.format(self.es_precision)) + '  ' + str('{:.4f}'.format(self.en_precision)) + '  ' + str('{:.4f}'.format(self.pt_precision))
        return classPrecision

    def outputClassRecalls(self):
        # eu-R, ca-R, gl-R, es-R, en-R and pt-R
        classRecall = str('{:.4f}'.format(self.eu_recall)) + '  ' + str('{:.4f}'.format(self.ca_recall)) + '  ' + str('{:.4f}'.format(self.gl_recall)) + '  '
        classRecall += str('{:.4f}'.format(self.es_recall)) + '  ' + str('{:.4f}'.format(self.en_recall)) + '  ' + str('{:.4f}'.format(self.pt_recall))
        return classRecall

    def outputClassF1(self):
         # eu-F, ca-F, gl-F, es-F, en-F and pt-F
        classF1 = str('{:.4f}'.format(self.eu_f1)) + '  ' + str('{:.4f}'.format(self.ca_f1)) + '  ' + str('{:.4f}'.format(self.gl_f1)) + '  '
        classF1 += str('{:.4f}'.format(self.es_f1)) + '  ' + str('{:.4f}'.format(self.en_f1)) + '  ' + str('{:.4f}'.format(self.pt_f1))
        return classF1



def main():
    print("Stats main.")

if __name__ == '__main__':
    main()
