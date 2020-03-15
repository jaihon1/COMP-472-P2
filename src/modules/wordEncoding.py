# (000 000 000 000 000 000 000 000 00) = (abc def ghi jkl mno pqr stu vwx yz)
#  max 15 characters in words
#  26 x 15 = 390 -> input vector
import numpy as np


class WordEncoding():
    def __init__(self, word):
        self.word = word
        self.encoded_values = []
        self.encoded_values_flatten = []
        self.encoded_values_capital = []
        self.max_word_length = 15
        self.word_length = len(word)


    def setup(self):
        self.encode()
        self.flatten()
        encoded_str = ''.join(str(value) for value in self.encoded_values_flatten)

        return encoded_str

    def encode(self):
        for char in self.word:
            encoded = self.encodeCharacter(char)
            self.encoded_values.append(encoded)

        if self.word_length < self.max_word_length:
            for i in range(self.max_word_length - self.word_length):
                encoded_vector = np.zeros((26,), dtype=int)
                self.encoded_values.append(encoded_vector)


    def flatten(self):
        for sublist in self.encoded_values:
            for item in sublist:
                self.encoded_values_flatten.append(item)

    def encodeCharacter(self, char):
        encoded_vector = np.zeros((26,), dtype=int)

        if char == 'a':
            encoded_vector[0] = 1
            return encoded_vector
        elif char == 'b':
            encoded_vector[1] = 1
            return encoded_vector
        elif char == 'c':
            encoded_vector[2] = 1
            return encoded_vector
        elif char == 'd':
            encoded_vector[3] = 1
            return encoded_vector
        elif char == 'e':
            encoded_vector[4] = 1
            return encoded_vector
        elif char == 'f':
            encoded_vector[5] = 1
            return encoded_vector
        elif char == 'g':
            encoded_vector[6] = 1
            return encoded_vector
        elif char == 'h':
            encoded_vector[7] = 1
            return encoded_vector
        elif char == 'i':
            encoded_vector[8] = 1
            return encoded_vector
        elif char == 'j':
            encoded_vector[9] = 1
            return encoded_vector
        elif char == 'k':
            encoded_vector[10] = 1
            return encoded_vector
        elif char == 'l':
            encoded_vector[11] = 1
            return encoded_vector
        elif char == 'm':
            encoded_vector[12] = 1
            return encoded_vector
        elif char == 'n':
            encoded_vector[13] = 1
            return encoded_vector
        elif char == 'o':
            encoded_vector[14] = 1
            return encoded_vector
        elif char == 'p':
            encoded_vector[15] = 1
            return encoded_vector
        elif char == 'q':
            encoded_vector[16] = 1
            return encoded_vector
        elif char == 'r':
            encoded_vector[17] = 1
            return encoded_vector
        elif char == 's':
            encoded_vector[18] = 1
            return encoded_vector
        elif char == 't':
            encoded_vector[19] = 1
            return encoded_vector
        elif char == 'u':
            encoded_vector[20] = 1
            return encoded_vector
        elif char == 'v':
            encoded_vector[21] = 1
            return encoded_vector
        elif char == 'w':
            encoded_vector[22] = 1
            return encoded_vector
        elif char == 'x':
            encoded_vector[23] = 1
            return encoded_vector
        elif char == 'y':
            encoded_vector[24] = 1
            return encoded_vector
        elif char == 'z':
            encoded_vector[25] = 1
            return encoded_vector



def main():
    print("WordEncoding main.")

if __name__ == '__main__':
    main()
