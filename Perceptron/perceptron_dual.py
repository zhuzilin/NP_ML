# Simple implementation for 1 layer perceptron
# dual learning algorithm
import numpy as np
import numpy.linalg as LA

def gram(x):
    return np.matmul(x, x.T)

class Perceptron:
    def __init__(self, dim=2, eta=1):
        self.dim = 2
        self.eta = eta
        self.alpha = None  # np.random.randn(dim)
        self.W = np.zeros(dim)  # np.random.randn(dim)
        self.b = np.zeros(1)  # np.random.randn()
        self.Gram = None
        
    def fit(self, x, y, detailed=False):
        self.alpha = np.zeros(y.shape)
        self.Gram = gram(x)
        i = 0
        cnt = 0
        epoch = 0
        finished = True
        
        while cnt != x.shape[0]:
            cnt += 1
            if y[i]*(np.dot(self.alpha*y, self.Gram[i, :])+self.b) <= 0:
                self.alpha[i] += self.eta
                self.b += self.eta*y[i]
                cnt = 0
                epoch += 1
                if detailed == True:
                    print("Epoch:", epoch, ": alpha", self.alpha, "b:", self.b)
                    
            i = (i+1)%x.shape[0]
        # Add a new axis to make the multiply column-wise
        self.W = (self.alpha*y)[..., None]*x
    
    # the predict and score part are the same as the primary one
    def predict(self, x):
        return np.sign((np.sum(self.W*x, axis=-1)+self.b))
        
    def score(self, x, y):
        if x.shape[-1] != self.dim:
            print("The input shape is incorrect!")
            return 0
        dis = np.abs(np.sum(self.W*x, axis=-1)+self.b)*y
        return -np.sum(dis*(dis<0))*1/LA.norm(self.W)
        
        
p = Perceptron()
p.fit(np.array([[3, 3], 
                [4, 3],
                [1, 1]]), np.array([1, 1, -1]), detailed=True)
print(p.predict(np.array([[3, 3], 
                          [4, 3],
                          [1, 1]])))