import numpy as np
import numpy.linalg as LA

# calculate ACF from l=0~l
def ACF(x, l):
    acf = np.zeros(l+1)
    for i in range(l+1):
        cov_matrix = np.cov(x[:len(x)-i], x[i:])
        acf[i] = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0]*cov_matrix[1, 1])
    return acf

# the sample PACF is computed via the 
# Durbin-Levinson recursive algorithm
def PACF(x, l, acf=None):
    if acf == None:
        acf = ACF(x, l)
    pacf = np.zeros([l+1, l+1])
    for k in range(1,l+1):
        pacf[k, k] = (acf[k] - np.dot(pacf[k-1, 1:k], acf[k-1:0:-1])) / \
                     (1 - np.dot(pacf[k-1, 1:k], acf[1:k]))
        for j in range(1,k):
            pacf[k, j] = pacf[k-1, j] - pacf[k, k]*pacf[k-1, k-j]
            
    return pacf

class AR:
    def __init__(self, p=0):
        self.p = p
        self.sigma_a = 0
        self.phi = None
        self.res = None
        
    def identifyP(self, x, method="PACF"):
        if method == "PACF":
            self.identifyP_PACF(x, epsilon=0.05)
    
    def identifyP_PACF(self, x, l=20, epsilon=0.05):
        pacf = PACF(x, l)
        for i in range(1, l+1):
            if np.abs(pacf[i, i]) < epsilon:
                self.p = i-1
                break
    
    def fit(self, x):
        """
        Just a linear regression using least square.
        """
        if self.p == 0: # if did not explicitly assign p
            self.identifyP(x)
        X = np.zeros([len(x)-self.p, self.p+1])
        Y = np.zeros([len(x)-self.p, 1])
        for i in range(len(x)-self.p):
            Y[i, 0] = x[i+self.p]
            X[i, 0] = 1
            X[i, 1:] = x[i:i+self.p] # Caution! Here the order of the input is not the order in the model
        self.phi = np.matmul(np.matmul(LA.inv(np.matmul(X.T, X)), X.T), Y)
        self.res = Y - np.matmul(X, self.phi)
        self.sigma_a = np.sqrt(np.sum(self.res*self.res)/(len(x)-2*self.p-1))
    
    def predict(self, x, step=1):
        X = np.zeros([1, self.p+1])
        X[0, 0] = 1
        if step == 1:
            return np.matmul(X, self.phi)
        else:
            pred = []
            for _ in range(step):
                pred.append(float(np.matmul(X, self.phi)))
                X[0, 1:-1] = X[0, 2:]
                X[0, -1] = pred[-1]
            return np.array(pred)
data = np.genfromtxt("./data/dgnp82.dat")

ar3 = AR(3)
ar3.fit(data)
print(ar3.phi)
print(ar3.sigma_a*ar3.sigma_a)
print(ar3.predict(data, step=8))
