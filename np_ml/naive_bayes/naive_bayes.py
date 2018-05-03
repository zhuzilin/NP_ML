from tqdm import tqdm
import random

class NaiveBayes:
    def __init__(self, laplace=1):
        self.laplace = 1
        self.data = None
        self.cnts = None
        self.total = None
        self.priors = None
        
    def fit(self, x, y, detailed=False):
        self.x = x
        self.y = y
        
    def predict(self, x, ys, priors=None, detailed=False):
        """ys is the type of output, e.x. [1, -1] or ['spam', 'ham']"""
        if self.priors is None:  # if no prior distribution is given, get it from data
            if priors is not None:
                self.priors = priors
            else :
                self.priors = {}
                for y in ys:
                    self.priors[y] = 0
                for y in self.y:
                    self.priors[y] += 1
        # initialize the counting dictionary
        # here we assume the order in x in insignificance.
        if self.cnts is None:
            self.cnts = {}
            self.total = {}
            for y in ys:
                self.cnts[y] = {}
                self.total[y] = 0
            for i in range(len(self.y)):
                assert self.y[i] in ys, "ys is wrong!"
                tmp_set = set()
                for element in self.x[i]:
                    if element not in tmp_set:
                        tmp_set.add(element)
                        if element not in self.cnts[self.y[i]]:
                            self.cnts[self.y[i]][element] = 1
                        else:
                            self.cnts[self.y[i]][element] += 1
                        self.total[y] += 1
        if not x or type(x[0]) != list:
            max_posterior = 0
            max_y = None
            for y in ys:
                posterior = 1
                for element in x:
                    if element in self.cnts[y]:
                        posterior *= (self.cnts[y][element]+self.laplace) / (self.priors[y]+len(self.cnts[y])*self.laplace)
                    else:
                        posterior *= self.laplace / (self.priors[y]+len(self.cnts[y])*self.laplace)
                posterior *= (self.priors[y]+self.laplace) / (len(self.y)+len(self.priors)*self.laplace)
                if detailed == True:
                    print("x:", x, "y:", y, "posterior:", posterior)
                if posterior > max_posterior:
                    max_posterior = posterior
                    max_y = y
            return max_y
        else:
            ans = []
            for i in tqdm(range(len(x)), ascii=True):
                ans.append(self.predict(x[i], ys, detailed=detailed))
            return ans
    def score(self, x, y):
        pass
        