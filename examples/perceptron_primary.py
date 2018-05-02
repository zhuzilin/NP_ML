import numpy as np
import csv

from np_ml import Perceptron, utils

if __name__ == '__main__':
    with open(r'..\data\iris.csv', newline='') as csvfile:
        data = np.array(list(csv.reader(csvfile)))

    # use binary classifier
    data = data[data[:, 4] != 'Iris-setosa']
    np.random.shuffle(data)
    x = np.array(data[:, :-1], dtype=np.float32)
    x = utils.transform(x, 2)
    y = data[:, -1]
    y[y == 'Iris-versicolor'] = 1
    y[y == 'Iris-virginica'] = -1
    y = np.array(y, dtype=np.int32)
    
    p = Perceptron(dim=x.shape[-1], eta=0.01, max_epoch=5000)
    p.fit(x, y, detailed=False)
    y_pred = p.predict(x)
    accuracy = np.sum(y_pred == y) / len(y)
    utils.plot_boundary(p, x, y, title="Perceptron", accuracy=accuracy)