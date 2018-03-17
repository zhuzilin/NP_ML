import numpy as np

# x > v or x < v
# y = 1 or -1
class TrivialClassification:
    def __init__(self):
        self.sign = None
        self.thres = 0
    
    def __str__(self):
        return self.sign + " than " + str(self.thres)
    
    def fit(self, x, y, w=None):
        if w is None:
            w = np.ones(len(y)) / len(y)
        data = zip(x, y, w)
        data = sorted(data, key=lambda s: s[0])
        [x, y, w] = zip(*data)
        y = np.array(y)
        w = np.array(w)
        correct = np.zeros((2, len(y))) # 0 row for x < v, 1 row for x >= v
        for i in range(len(y)):
            w_front = w[:i]
            w_back  = w[i:]
            correct[0, i] += np.sum(w_front[y[:i] == 1]) + np.sum(w_back[y[i:] == -1])
            correct[1, i] += np.sum(w_front[y[:i] == -1]) + np.sum(w_back[y[i:] == 1])
        idx = np.argmax(correct, axis=1)
        if correct[0, int(idx[0])] > correct[1, int(idx[1])]:
            self.sign = "smaller"
            self.thres = x[idx[0]]
        else:
            self.sign = "equal to or bigger"
            self.thres = x[idx[1]]
            
    def predict(self, x):
        if self.sign == "smaller":
            return (x < self.thres)*2-1
        else:
            return (x >= self.thres)*2-1
    
    def score(self, x, y, w=None): # the wrong percent
        if w is None:
            w = np.ones(len(y)) / len(y)
        return 1 - np.sum(w[self.predict(x) == y])

class AdaBoost:
    def __init__(self, weak_learner, epsilon=0.01):
        self.weak_learner_class = weak_learner
        self.weak_learners = []
        self.alphas = []
        self.epsilon = 0.01
        
    @staticmethod
    def calcAlpha(e):
        return 0.5*np.log((1-e)/e)
        
    def fit(self, x, y, detailed=False):
        w = np.ones(len(y)) / len(y)
        score = 1
        while score > self.epsilon:
            wl = self.weak_learner_class()
            wl.fit(x, y, w)
            alpha = AdaBoost.calcAlpha(wl.score(x, y, w))
            self.alphas.append(alpha)
            self.weak_learners.append(wl)
            w = w*np.exp(-alpha*y*self.predict(x))
            w = w/np.sum(w)
            score = self.score(x, y)
            if detailed:
                print("Weak learner:", wl, ", alpha:", alpha)
                print(score)
    def predict(self, x):
        ans = np.zeros(x.shape[0])
        for i in range(len(self.alphas)):
            ans += self.weak_learners[i].predict(x)*self.alphas[i]
        return (ans > 0)*2-1
        
    def score(self, x, y):
        return 1 - np.sum(self.predict(x) == y)/len(y)
        
tc = TrivialClassification()
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])

adb = AdaBoost(TrivialClassification)
adb.fit(x, y, detailed=True)
print(adb.predict(x))