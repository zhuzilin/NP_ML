# LDA model with Gibbs Sampling
# the implementation is based on 
# Darling W M. A theoretical and practical implementation tutorial on topic modeling and gibbs sampling, 2011.
# Gregor Heinrich, Parameter estimation for text analysis, 2004
# Many variables may have sparse property which may help optimize 
# computation.
import numpy as np
import random
from collections import OrderedDict
import os

class Documents:
    def __init__(self, data=None, dir=None):
        if data is not None:
            self.documents = []
            self.ndoc = len(data)
            self.dict = {}
            self.reverse_dict = {}
            self.nword = 0
            for document in data:
                self.documents.append(np.zeros(len(document)))
                for i in range(len(document)):
                    word = document[i]
                    if not word in self.dict:
                        self.dict[word] = self.nword
                        self.reverse_dict[self.nword] = word
                        self.nword += 1
                    self.documents[-1][i] = self.dict[word]
        else: # input a directory
            pass
                
class LDA:
    def __init__(self, K=2, alpha=0.1, beta=0.1):
        self.K = K
        self.alpha=0.1
        self.beta=0.1
        # the following variable is correspond to the link provided
        self.n_d_k = None
        self.n_k_word = None
        self.n_k = None
        self.phi = None
        self.theta = None
        
    def fit(self, data, iter_times=100, detailed=False):
        # initial variables
        self.n_d_k = np.zeros((data.ndoc, self.K))
        self.n_k_word = np.zeros((self.K, data.nword))
        self.n_k = np.zeros(self.K)
        self.n_d = np.zeros(data.ndoc)
        self.p = np.zeros(self.K) # is not normalized
        z = [np.zeros(len(document)) for document in data.documents] # Here we will only use the shape
        for d in range(data.ndoc):
            document = data.documents[d]
            for w in range(len(document)):
                word = document[w]
                k = np.random.randint(self.K)
                z[d][w] = k
                self.n_d_k[d, k] += 1
                self.n_k_word[k, int(word)] += 1
                self.n_k[k] += 1
                self.n_d[d] += 1
        # Gibbs Sampling
        for epoch in range(iter_times):
            if detailed:
                print("Epoch:", epoch)
            for d in range(data.ndoc):
                document = data.documents[d]
                for w in range(len(document)):
                    word = document[w]
                    k = int(z[d][w])
                    self.n_d_k[d, k] -= 1
                    self.n_k_word[k, int(word)] -= 1
                    self.n_k[k] -= 1
                    self.n_d[d] -= 1
                    self.p = (self.n_d_k[d, :]+self.alpha)*(self.n_k_word[:, int(word)]+self.beta)/(self.n_k + self.beta*data.nword)
                    p = np.random.uniform(0, np.sum(self.p))
                    # print("p:", p)
                    # print("self.p:", self.p)
                    for k in range(self.K):
                        if p <= self.p[k]:
                            z[d][w] = k
                            self.n_d_k[d, int(k)] += 1
                            self.n_k_word[k, int(word)] += 1
                            self.n_k[k] += 1
                            self.n_d[d] += 1
                            break
                        else:
                            p -= self.p[k]
        self.phi = np.zeros((self.K, data.nword))
        for k in range(self.K):
            self.phi[k, :] = (self.n_k_word[k, :] + self.beta)/(self.n_k[k] + self.beta)
        self.theta = np.zeros((data.ndoc, self.K))
        for d in range(data.ndoc):
            self.theta[d, :] = (self.n_d_k[d, :] + self.alpha)/(self.n_d[d] + self.alpha)
        
if __name__ == '__main__':
    print("data1")
    data = [["apple", "orange", "banana"], 
            ["apple", "orange"],
            ["orange", "banana"],
            ["cat", "dog"], 
            ["dog", "tiger"], 
            ["tiger", "cat"]]

    docs = Documents(data=data)
    lda = LDA()
    lda.fit(docs)
    print(docs.reverse_dict)
    print(lda.theta)
    print(lda.phi)
    print("")

    print("data2")
    data = [[1, 2, 3, 1, 2], 
            [1, 4, 5, 4, 4],
            [1, 4, 2, 5, 5, 4],
            [1, 3, 3, 2, 3],
            [1, 1, 3, 2, 2]]
    docs = Documents(data=data)
    lda.fit(docs)
    print(docs.reverse_dict)
    print(lda.theta)
    print(lda.phi)