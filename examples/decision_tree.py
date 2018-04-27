import numpy as np
import csv

from np_ml import DecisionTree, utils

if __name__ == '__main__':
    with open(r'..\data\tic_tac_toe.csv', newline='') as csvfile:
        data = np.array(list(csv.reader(csvfile)))
    np.random.shuffle(data)
    train = data[:int(data.shape[0]*0.8), :]
    test = data[int(data.shape[0]*0.8):, :]
    train_x = train[:, :-1]
    train_y = train[:, -1]
    train_y = (train_y == "positive")
    test_x = test[:, :-1]
    test_y = test[:, -1]
    test_y = (test_y == "positive")
                  
    print("use ID3: ")
    dt = DecisionTree(max_depth=1)
    dt.fit(train_x, train_y, type="ID3", detailed=False)
    y_pred = dt.predict(test_x)
    accuracy = np.sum(y_pred == test_y) / len(test_y)
    print("tree struct: ")
    print(dt.root)
    print("accuracy: "+str(accuracy))
    
    print("use CART: ")
    dt = DecisionTree(max_depth=1)
    dt.fit(train_x, train_y, type="CART", detailed=False)
    y_pred = dt.predict(test_x)
    accuracy = np.sum(y_pred == test_y) / len(test_y)
    print("tree struct: ")
    print(dt.root)
    print("accuracy: "+str(accuracy))