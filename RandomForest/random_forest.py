import numpy as np
import csv
from decision_tree import DecisionTree
from collections import Counter

class RandomForest:
    def __init__(self, num):
        self.num = num
        self.dts = []
        for _ in range(num):
            self.dts.append(DecisionTree())
    def fit(self, x, y, detailed=False):
        num_attribute = x.shape[1]
        gap = num_attribute // self.num
        for i in range(self.num-1):
            self.dts[i].fit(x[:, i*gap:(i+1)*gap], y, detailed=detailed)
        self.dts[-1].fit(x[:, (self.num-1)*gap:], y, detailed=detailed)
    
    def predict(self, x):
        votes = []
        num_attribute = x.shape[1]
        gap = num_attribute // self.num
        for i in range(self.num-1):
            votes.append(self.dts[i].predict(x[:, i*gap:(i+1)*gap]))
        votes.append(self.dts[-1].predict(x[:, (self.num-1)*gap:]))
        # print(votes)
        return np.sum(np.array(votes), axis=0) > len(votes) / 2
        
    def evaluate(self, x, y):
        y_pred = self.predict(x)
        return np.sum(np.array(y) == np.array(y_pred)) / len(y)
        
with open('tic_tac_toe.csv', newline='') as csvfile:
    data = np.array(list(csv.reader(csvfile)))
# data = np.genfromtxt('tic_tac_toe.csv', delimiter=',', dtype=None)
np.random.shuffle(data)
train = data[:int(data.shape[0]*0.8), :]
test = data[int(data.shape[0]*0.8):, :]
train_x = train[:, :-1]
train_y = train[:, -1]
train_y = (train_y == "positive")
test_x = test[:, :-1]
test_y = test[:, -1]
test_y = (test_y == "positive")
'''
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
'''
              
rf = RandomForest(2)
rf.fit(train_x[:, :], train_y, detailed=False)
print(rf.evaluate(test_x[:, :], test_y))