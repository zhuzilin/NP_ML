# SVM using SMO algorithm
import numpy as np
import numpy.linalg as LA

np.set_printoptions(precision=2)

class SVM:
    def __init__(self):
        self.alpha = None
        self.b = 0
        self.kernel = None
        self.x = None
        self.y = None
        
    def g(self, x0): # TODO: find a vectorized way 
            ans = 0
            for i in range(len(self.y)):
                ans += self.alpha[i]*self.y[i]*self.kernel(self.x[i, :], x0)
            return ans + self.b
        
    def fit(self, x, y, kernel=None, C=10., epsilon=0.1, detailed=False):
        # initialization
        self.x = x
        self.y = y
        if kernel is None: # default using linear kernel
            self.kernel = lambda x1, x2: np.matmul(x1, x2.T)
        else:
            self.kernel = kernel
        self.alpha = np.zeros(y.shape)
        E = -y
        epoch = 0
        while True: #cnt<20:
            if detailed:
                epoch += 1
                print("Epoch:", epoch)
                if kernel is None:
                    print("    W      :", np.sum((self.alpha*y)[..., None]*x, axis=0))
                else:
                    print("    alpha  :", self.alpha)
                print("    b      :", self.b)
                print("    E      :", E)
            # check if all conditions are met and if not find alpha1
            i1 = -1 # the index for alpha1
            worst = epsilon
            for i in range(len(y)):
                if self.alpha[i] == 0:
                    if y[i]*self.g(x[i, :]) < 1-worst:
                        worst = 1 - y[i]*self.g(x[i, :])
                        i1 = i
                elif self.alpha[i] == C:
                    if y[i]*self.g(x[i, :]) > 1+epsilon:
                        worst = y[i]*self.g(x[i, :]) - 1
                        i1 = i
                else:
                    if abs(y[i]*self.g(x[i, :])-1) > epsilon:
                        worst = abs(y[i]*self.g(x[i, :])-1)
                        i1 = i
            if i1 == -1:
                break
            # find alpha2
            # Here to make it easy, I do not save the E value, but calculate it each time
            i2 = -1
            max_abs = -1
            for i in range(len(y)):
                if i == i1:
                    continue
                if abs(E[i] - E[i1]) > max_abs:
                    i2 = i
                    max_abs = abs(E[i] - E[i1])
            # update
            if y[i1] != y[i2]:
                L = max(0., self.alpha[i2]-self.alpha[i1])
                H = min(C, C+self.alpha[i2]-self.alpha[i1])
            else:
                L = max(0., self.alpha[i2]+self.alpha[i1]-C)
                H = min(C, self.alpha[i2]+self.alpha[i1])
            # update alpha1, alpha2, b
            eta = self.kernel(x[i1, :], x[i1, :]) + self.kernel(x[i2, :], x[i2, :]) - 2*self.kernel(x[i1, :], x[i2, :])
            alpha2_new = min(H, max(L, self.alpha[i2] + y[i2]*(E[i1] - E[i2]) / eta))
            alpha1_new = self.alpha[i1] + y[i1]*y[i2]*(self.alpha[i2]-alpha2_new)
            b1_new = -E[i1] + y[i1]*self.kernel(x[i1, :], x[i1, :])*(self.alpha[i1] - alpha1_new) + \
                              y[i2]*self.kernel(x[i2, :], x[i1, :])*(self.alpha[i2] - alpha2_new) + self.b
            b2_new = -E[i2] + y[i1]*self.kernel(x[i1, :], x[i2, :])*(self.alpha[i1] - alpha1_new) + \
                              y[i2]*self.kernel(x[i2, :], x[i2, :])*(self.alpha[i2] - alpha2_new) + self.b
            b_new = (b1_new + b2_new)/2
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new
            # update E
            for i in range(len(y)):
                E[i] = self.g(x[i, :]) - y[i]
            
    def predict(self, x):
        if x.ndim == 1:
                return np.sign(self.g(x))
        else:
            ans = []
            for i in range(x.shape[0]):
                ans.append(self.predict(x[i, :]))
            return ans
        
if __name__ == '__main__':
    # solution should be 0.5x + 0.5y - 2 = 0
    x = np.array([[1., 1.], 
                  [4., 3.],
                  [3., 3.]], np.float32)
                  
    y = np.array([-1., 1., 1.], np.float32)

    svm = SVM()
    svm.fit(x, y, detailed=True)
    print(svm.predict(x))

    # more complex kernel
    x = np.array([[0, 0], 
                  [0, 1],
                  [1, 1],
                  [1, 0],
                  [0, 0.5],
                  [0.5, 0],
                  [0.5, 1],
                  [1, 0.5]], np.float32)
    y = np.array([-1, -1, -1, -1, 1, 1, 1, 1], np.float32)
    gaussian = lambda x1, x2: np.exp(-LA.norm(x1-x2)/2)
    svm = SVM()
    svm.fit(x, y, kernel=gaussian, detailed=True)
    print(svm.predict(x))