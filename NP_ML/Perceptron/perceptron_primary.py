# Simple implementation for 1 layer perceptron
# primary learning algorithm
import numpy as np
import numpy.linalg as LA

class Perceptron:
    def __init__(self, dim=2, eta=1, max_epoch=None):
        self.dim = dim
        self.eta = eta
        self.W = np.zeros(dim)  # np.random.randn(dim)
        self.b = np.zeros(1)  # np.random.randn()
        self.max_epoch = 1000
        
    def fit(self, x, y, detailed=False):
        i = 0
        cnt = 0
        epoch = 0
        finished = True
        
        while cnt != x.shape[0] and (self.max_epoch is None or epoch < self.max_epoch):
            cnt += 1
            if y[i]*(np.sum(self.W*x[i, :], axis=-1)+self.b) <= 0:
                self.W += self.eta*y[i]*x[i, :]
                self.b += self.eta*y[i]
                cnt = 0
                epoch += 1
                if detailed == True:
                    print("Epoch:", epoch, ": W", self.W, "b:", self.b)
                    
            i = (i+1)%x.shape[0]
    
    def predict(self, x):
        return np.sign((np.sum(self.W*x, axis=-1)+self.b))
        
    def score(self, x, y):
        if x.shape[-1] != self.dim:
            print("The input shape is incorrect!")
            return 0
        dis = np.abs(np.sum(self.W*x, axis=-1)+self.b)*y
        return -np.sum(dis*(dis<0))*1/LA.norm(self.W)
        
if __name__ == '__main__':
    p = Perceptron()
    p.fit(np.array([[3, 3], 
                    [4, 3],
                    [1, 1]]), np.array([1, 1, -1]), detailed=True)
    print(p.predict(np.array([[3, 3], 
                              [4, 3],
                              [1, 1]])))