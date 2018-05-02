import numpy as np
from np_ml import AdaBoost, TrivialClassification

if __name__ == '__main__':
    print("--------------------------------------------------------")
    print("AdaBoost simple example!")
    print("example in Statistical Learning Method(《统计学习方法》)")
    print("--------------------------------------------------------")
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
    print("x: {}".format(x))
    print("y: {}".format(y))
    print("")
    adb = AdaBoost(TrivialClassification)
    adb.fit(x, y, detailed=True)
    print("y_pred: {}".format(adb.predict(x)))
    