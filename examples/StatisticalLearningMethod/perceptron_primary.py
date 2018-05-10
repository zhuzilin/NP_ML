import numpy as np
from np_ml import Perceptron

if __name__ == '__main__':
    print("--------------------------------------------------------")
    print("Perceptron simple example!")
    print("example in Statistical Learning Method(《统计学习方法》)")
    print("--------------------------------------------------------")
    p = Perceptron()
    x = np.array([[3, 3], 
                 [4, 3],
                 [1, 1]])
    y = np.array([1, 1, -1])
    print("x: ")
    print(x)
    print("y: ")
    print(y)
    print("")
    p.fit(x, y, detailed=True)
    print("y_pred: ")
    print(p.predict(np.array([[3, 3], 
                              [4, 3],
                              [1, 1]])))