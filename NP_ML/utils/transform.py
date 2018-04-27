import numpy as np

# PCA
def transform(x, dim):
    e_val, e_vec = np.linalg.eig(np.cov(x, rowvar=False))
    idx = e_val.argsort()[::-1]
    e_vec = e_vec[:, idx][:, :dim]
    return x.dot(e_vec)