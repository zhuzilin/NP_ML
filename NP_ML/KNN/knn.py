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
        self.heap = []
        
    def predict(self, x, k=5, similarity="euclidean", detailed=False):
        if x.ndim == 1:
            if similarity == "euclidean":
                for i in range(len(self.train_y)):
                    heappush(self.heap, (-LA.norm(self.train_x[i, :]-x), list(self.train_x[i, :]), self.train_y[i]))
                    if len(self.heap) > k:
                        heappop(self.heap)
            if detailed:
                print("For",x,", the",k,"nearest neighbor are:")
                for n in self.heap[::-1]:
                    print("x:", n[1], ", y: ", n[2])
            [_, _, votes] = zip(*self.heap)
            self.heap = []
            if votes.count(1) > votes.count(-1):
                return 1
            else:
                return -1
        else:
            return [self.predict(x[i, :], detailed=detailed) for i in range(x.shape[0])]
            
if __name__ == '__main__':
    x = np.array([[0, 0],
                  [0, 0.3],
                  [0.1, 0.2],
                  [0.2, 0.4],
                  [0, 1],
                  [0.3, 0.8],
                  [0.2, 0.9],
                  [1, 0],
                  [1.1, 0.1],
                  [0.7, 0.3],
                  [1, 1],
                  [0.9, 0.9],
                  [0.8, 0.7]])
    y = np.array([1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1])

    knn = KNN()
    knn.fit(x, y)
    print(knn.predict(np.array([[0.1, 0.3],
                       [0.2, 0.8],
                       [0.9, 0.1],
                       [1.2, 1.5]]), detailed=True))