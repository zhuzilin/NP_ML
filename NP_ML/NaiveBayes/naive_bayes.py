import numpy as np

class NaiveBayes:
    def __init__(self, laplace=1):
        self.laplace = 1
        
    def fit(self, x, y, detailed=False):
        self.data = np.concatenate((x, y[..., None]), axis=-1)
        
    def predict(self, x, ys=[-1, 1], detailed=False):
        priors = {}
        for y in ys:
            priors[y] = self.data[self.data[:, -1] == y].shape[0]
        if x.ndim == 1:
            max_posterior = 0
            max_y = None
            for y in ys:
                data_y = self.data[self.data[:, -1] == y]
                posterior = 1
                for i in range(len(x)):
                    unique, cnts = np.unique(data_y[:, i], return_counts=True)
                    cnts = dict(zip(unique, cnts))
                    if x[i] in cnts:
                        posterior *= (cnts[x[i]]+self.laplace) / (priors[y]+len(cnts)*self.laplace)
                    else:
                        posterior = 0
                        break
                posterior *= (priors[y]+self.laplace) / (self.data.shape[0]+len(priors)*self.laplace)
                if detailed == True:
                    print("x:", x, "y:", y, "posterior:", posterior)
                if posterior > max_posterior:
                    max_posterior = posterior
                    max_y = y
            return max_y
        else:
            ans = []
            for i in range(x.shape[0]):
                ans.append(self.predict(x[i, :], detailed=detailed))
            return ans
    def score(self, x, y):
        pass
        
# To make it easier, set S as 1, M as 2, L as 3
x = np.array([[1, 0],
              [1, 1],
              [1, 1],
              [1, 0],
              [1, 0],
              [2, 0],
              [2, 1],
              [2, 1],
              [2, 2],
              [2, 2],
              [3, 2],
              [3, 1],
              [3, 1],
              [3, 2],
              [3, 2]])
y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
nb = NaiveBayes()
nb.fit(x, y)
print(nb.predict(np.array([[2, 0]]), detailed=True))