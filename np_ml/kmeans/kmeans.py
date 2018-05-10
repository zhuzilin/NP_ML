import numpy as np
import scipy

class KMeans:
    def __init__(self, k=2, eps=1e-5):
        self.k = k
        self.eps = eps
        self.centers = None

    def fit(self, x, detailed=False):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        self.centers = x[np.random.randint(x.shape[0], size=self.k)]
        flag = True
        label = None
        while flag:
            flag = False
            dist_matrix = scipy.spatial.distance.cdist(x, self.centers)
            label = np.argmin(dist_matrix, axis=1)
            for i in range(self.k):
                new_center = x[label == i].mean()
                if np.linalg.norm(new_center-self.centers[i]) > self.eps:
                    flag = True
                self.centers[i] = new_center
            if detailed:
                for i in range(self.k):
                    print("centers:")
                    print(self.centers[i])
                    print("")
        return label

    def predict(self, x):
        dist_matrix = scipy.spatial.distance.cdist(x, self.centers)
        label = np.argmin(dist_matrix, axis=1)
        return label