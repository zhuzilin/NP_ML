import csv
import numpy as np
from np_ml import KNN, utils

if __name__ == '__main__':
    with open(r'..\data\iris.csv', newline='') as csvfile:
        data = np.array(list(csv.reader(csvfile)))

    # use binary classifier
    np.random.shuffle(data)
    x = np.array(data[:, :-1], dtype=np.float32)
    x = utils.transform(x, 2)
    y = data[:, -1]
    y[y == 'Iris-versicolor'] = 0
    y[y == 'Iris-virginica'] =  1
    y[y == 'Iris-setosa'] =  2
    y = np.array(y, dtype=np.int32)
    knn = KNN()
    knn.fit(x, y)
    utils.plot_boundary(knn, x, y, title="K Nearest Neighbor")