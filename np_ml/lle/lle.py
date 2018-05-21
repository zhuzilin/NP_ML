"""LLE Algorithm

    for detail please see: https://cs.nyu.edu/~roweis/lle/algorithm.html
"""
import numpy as np
import numpy.linalg as LA
import scipy

class LLE:
    def __init__(self):
        pass

    def fit(self, x, k=10, dim=2):
        x = np.array(x)
        assert x.shape[0] > k, "too much neighbor selected!"
        dist = scipy.spatial.distance.cdist(x, x)
        nn = np.argsort(dist, axis=1)[:, 1:k+1]
        W = np.zeros(dist.shape)
        for i in range(x.shape[0]):
            Z = x[nn[i]]  # knn for x[i]
            Z -= x[i]
            C = np.matmul(Z, Z.T)
            if k > x.shape[1]:
                C += 1e-3 * np.trace(C)*np.identity(k)
            w = np.matmul(LA.inv(C), np.ones((k, 1)))
            W[i, nn[i]] = w.flatten() / np.sum(w)
        M = np.matmul((np.identity(x.shape[0])-W).T, (np.identity(x.shape[0])-W))
        e_val, e_vec = LA.eig(M)
        idx = e_val.argsort()
        e_vec = e_vec[:, idx][:, 1:dim+1]
        
        return e_vec