"""
Use Naive Bayes to discriminate spam e-mail.

Since we need to preprocess text, we import ntlk here.
"""
import csv
import random
import numpy as np
import string
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords

from np_ml import NaiveBayes

if __name__ == '__main__':
    with open(r'..\data\spam.csv', encoding='latin-1') as csvfile:
        data = list(csv.reader(csvfile))
    print("preprocessing data...")
    data = data[1:]
    random.shuffle(data)
    y = [email[0] for email in data]
    x_pre = [email[1] for email in data]
    x = []
    
    translator = str.maketrans('', '', string.punctuation)
    # translator = str.maketrans(dict.fromkeys(string.punctuation))
    for text in tqdm(x_pre, ascii=True):
        text = text.translate(translator)
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        # tokens = [word for word in tokens if word not in stopwords.words('english')]
        x.append(tokens)
    
    train_x = x[:int(len(y)*0.8)]
    train_y = y[:int(len(y)*0.8)]
    
    test_x = x[int(len(y)*0.8):]
    test_y = y[int(len(y)*0.8):]
    print("finish preprocessing data.")
    print("")
    
    nb = NaiveBayes()
    nb.fit(train_x, train_y)
    accuracy = np.sum(np.array(nb.predict(test_x, ys=['ham', 'spam'])) == np.array(test_y)) / len(test_y)
    print("accuracy: ", accuracy)
    
    print("two example:")
    example_ham= 'Po de :-):):-):-):-). No need job aha.'
    print("example ham: ")
    print(example_ham)
    example_ham = example_ham.lower()
    example_ham = nltk.word_tokenize(example_ham.translate(translator))
    print("predict result: ")
    print(nb.predict(example_ham, ys=['ham', 'spam']))
    print("")
    
    example_spam= 'u r a winner U ave been specially selected 2 receive 澹1000 cash or a 4* holiday (flights inc) speak to a live operator 2 claim 0871277810710p/min (18 )'
    print("example spam: ")
    print(example_spam)
    example_spam = example_spam.lower()
    example_spam = nltk.word_tokenize(example_spam.translate(translator))
    print("predict result: ")
    print(nb.predict(example_spam, ys=['ham', 'spam']))