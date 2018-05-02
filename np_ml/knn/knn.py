# As far as my research, all kinds of approximate way would degrade to linear scan.
# Therefore, I only implement the linear scan.
# For more advanced data structures and algorithms, 
# see here: https://stackoverflow.com/questions/5751114/nearest-neighbors-in-high-dimensional-data
import numpy as np
import numpy.linalg as LA
from heapq import heappush, heappop

class KNN:
    def __init__(self):
        pass
    
    def fit(self, x, y):
        self.train_x = x
        self.train_y = y
        
    def predict(self, x, k=5, similarity="euclidean", detailed=False):
        heap = []
        if x.ndim == 1:
            if similarity == "euclidean":
                for i in range(len(self.train_y)):
                    heappush(heap, (-LA.norm(self.train_x[i, :]-x), list(self.train_x[i, :]), self.train_y[i]))
                    if len(heap) > k:
                        heappop(heap)
            if detailed:
                print("For {}, the {} nearest neighbor are:".format(x, k))
                for n in heap[::-1]:
                    print("x: {}, y: {}".format(n[1], n[2]))
            [_, _, votes] = zip(*heap)
            return max(set(votes), key=votes.count)
        else:
            return np.array([self.predict(x[i, :], detailed=detailed) for i in range(x.shape[0])])