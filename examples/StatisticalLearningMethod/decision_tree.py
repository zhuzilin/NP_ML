import numpy as np
from np_ml import DecisionTree

if __name__ == "__main__":
    print("--------------------------------------------------------")
    print("DecisionTree simple example!")
    print("example in Statistical Learning Method(《统计学习方法》)")
    print("--------------------------------------------------------")
    x = np.array([["young",    False, False, "ordinary"],
                  ["young",    False, False, "good"],
                  ["young",    True,  False, "good"],
                  ["young",    True,  True,  "ordinary"],
                  ["young",    False, False, "ordinary"],
                  ["mid-life", False, False, "ordinary"],
                  ["mid-life", False, False, "good"],
                  ["mid-life", True,  True,  "good"],
                  ["mid-life", False, True,  "very good"],
                  ["mid-life", False, True,  "very good"],
                  ["old",      False, True,  "very good"],
                  ["old",      False, True,  "good"],
                  ["old",      True,  False, "good"],
                  ["old",      True,  False, "very good"],
                  ["old",      False, False, "ordinary"]])

    y = np.array([False,
                  False,
                  True,
                  True,
                  False,
                  False,
                  False,
                  True,
                  True,
                  True,
                  True,
                  True,
                  True,
                  True,
                  False])
    print("x: ")
    print(x)
    print("y: ")
    print(list(y))
    print("")
    dt = DecisionTree(max_depth=-1)
    dt.fit(x, y, type="C4.5", detailed=True)  # here we can change type to "ID3", "C4.5" or "CART"
    print("The result tree:")
    print(dt.root)
    print("y_pred: ")
    print(dt.predict(x))