import numpy as np
from np_ml import NaiveBayes

if __name__ == '__main__':
    print("--------------------------------------------------------")
    print("Naive Bayes simple example!")
    print("example in Statistical Learning Method(《统计学习方法》)")
    print("--------------------------------------------------------")
    # To make it easier, set S as 1, M as 2, L as 3
    x = [[1, 0],
                  [1, 1],
                  [1, 1],
                  [1, 0],
                  [1, 0],
                  [2, 0],
                  [2, 1],
                  [2, 1],
                  [2, 2],
                  [2, 2],
                  [3, 2],
                  [3, 1],
                  [3, 1],
                  [3, 2],
                  [3, 2]]
    y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    print("x: ")
    print(x)
    print("y: ")
    print(y)
    print("")
    nb = NaiveBayes()
    nb.fit(x, y)
    
    print("x_pred: {}".format(np.array([[2, 0]])))
    print("y_pred: {}".format(nb.predict(np.array([[2, 0]]), detailed=True)))