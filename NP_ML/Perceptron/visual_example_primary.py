import numpy as np
import csv
from perceptron_primary import Perceptron
import matplotlib.pyplot as plt
from ..utils.transform import transform

if __name__ == '__main__':
    with open(r'..\data\iris.csv', newline='') as csvfile:
        data = np.array(list(csv.reader(csvfile)))

    # use binary classifier
    data = data[data[:, 4] != 'Iris-setosa']
    x = np.array(data[:, :-1], dtype=np.float32)
    y = data[:, -1]
    y[y == 'Iris-versicolor'] = 1
    y[y == 'Iris-virginica'] = -1
    y = np.array(y, dtype=np.int32)

    p = Perceptron(dim=x.shape[-1], eta=0.01, max_epoch=1000)
    p.fit(x, y, detailed=False)
    print(np.sum(p.predict(x) == y) / len(y))

    t_x = transform(x, 2)
    plt.figure()
    color = ((y == p.predict(x))+1) / 3
    plt.scatter(t_x[:, 0], t_x[:, 1], c=color)
    lx = 5
    plt.show()