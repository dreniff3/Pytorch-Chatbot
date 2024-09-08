'''
    This module...
'''

import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

# download pre-trained tokenizer so .word_tokenize works
# nltk.data.path = [r'C:\Users\Don Reniff\AppData\Roaming\nltk_data']
# nltk.download('punkt')

# create stemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    '''
        Splits a sentence into meaningful units.
    '''
    return nltk.word_tokenize(sentence)


def stem(word):
    '''
        Generates the root form of words (using a crude heuristic that chops
        the end of words).
    '''
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    '''
        sentence = ["hello", "how", "are", "you"]
        words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        bag   = [ 0,      1,     0,    1,     0,      0,       0 ]
    '''
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)

    for index, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[index] = 1.0

    return bag


# s = ["hello", "how", "are", "you"]
# w = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
# bag = bag_of_words(s, w)
# print(bag)  # [0. 1. 0. 1. 0. 0. 0.]

# s = "How long does shipping take?"
# print(s)
# s = tokenize(s)
# print(s)

# words = ["organize", "organizes", "organizing"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)
