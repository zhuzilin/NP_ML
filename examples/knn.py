import numpy as np
from np_ml import KNN

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
                       
    np.random()