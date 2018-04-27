import numpy as np
import csv

from np_ml import utils

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
    
    t_x = utils.transform(x, 2)
    color = (y+2) / 5
    utils.plot(t_x, color, title="Ground Truth")