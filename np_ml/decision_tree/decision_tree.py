import numpy as np

def entropy(col):
    _, cnts = np.unique(col, return_counts=True)
    cnts = np.array(cnts)/len(col)
    cnts[cnts!=0] = cnts[cnts!=0]*np.log2(cnts[cnts!=0])
    return -np.sum(cnts)

# For ID3
def calcInforGain(col_x, col_y):
    HD = entropy(col_y)
    HDA = 0
    unique = np.unique(col_x)
    for key in unique:
        HDA += entropy(col_y[col_x == key])
    return HD - HDA, unique

# For C4.5
def calcInforGainRatio(col_x, col_y):
    HD = entropy(col_y)
    HDA = 0
    unique = np.unique(col_x)
    for key in unique:
        HDA += entropy(col_y[col_x == key])
    return (HD - HDA)/entropy(col_x), unique
    
# For CART
def Gini(col):
    unique, cnts = np.unique(col, return_counts=True)
    cnts = np.array(cnts)/len(col)
    return 1 - np.sum(cnts ** 2)
    
def findMinGini(col_x, col_y):
    unique, cnts = np.unique(col_x, return_counts=True)
    cnts = dict(zip(unique, cnts))
    min_gini = 1
    min_key = None
    for key, cnt in cnts.items():
        gini = cnt/len(col_y)*Gini(col_y[col_x == key]) + (1-cnt/len(col_y))*Gini(col_y[col_x != key])
        if gini < min_gini:
            min_gini = gini
            min_key = key
    return min_gini, min_key
    
class Node:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.children = []
        
    def __str__(self, indent=0):
        ans = ""
        if not self.children:
            ans = str(self.key) + ": " + str(self.val) + ""
        else:
            ans += str(self.key) + ": " + str(self.val) + "("
            for child in self.children:
                ans += str(child) + ", "
            ans += ")"
        return ans
    
    def addChild(self, key, val):
        self.children.append(Node(key, val))
        return self.children[-1]
    
class DecisionTree:
    def __init__(self, epsilon=0):
        self.root = Node("root", 0)
        self.epsilon = epsilon
        self.type = None
        
    def fit(self, x, y, type="CART", detailed=False):
        self.type = type
        if type == "CART":
            self.CARTgenerate(x, y, self.root, detailed)
        else:
            self.generate(x, y, self.root, type, detailed)
    
    def generate(self, x, y, root, detailed):
        # if empty
        if x.size == 0:
            return
        # if all left are the same kind
        if np.all(y == True) or np.all(y == False):
            root.addChild("leaf", y[0])
            return
        # if all the feature are the same, use the popular one
        if np.all(x == x[0,:]):
            unique, cnts = np.unique(y, return_counts=True)
            cnts = dict(zip(unique, cnts))
            root.addChild("leaf", cnts[True] > cnts[False])
            return 
        
        max_gain = 0
        max_feature = -1
        max_feature_vals = None
        
        for i in range(x.shape[-1]):
            if type=="ID3":
                gain, feature_vals = calcInforGain(x[:, i], y)
            elif type=="C4.5":
                gain, feature_vals = calcInforGainRatio(x[:, i], y)
            if gain > max_gain:
                max_gain = gain
                max_feature = i
                max_feature_vals = feature_vals
        if max_gain < self.epsilon:
            return
        else:
            for val in max_feature_vals:
                child = root.addChild(max_feature, val)
                self.generate(np.delete(x[x[:, max_feature]==val], max_feature, axis=-1), y[x[:, max_feature]==val], child, type, detailed)
    
    def CARTgenerate(self, x, y, root, detailed, min_gini_old=1):
        # if empty
        if x.size == 0:
            return
        # if all left are the same kind
        if np.all(y == True) or np.all(y == False):
            root.addChild("leaf", y[0])
            return
        # if all the feature are the same, use the popular one
        if np.all(x == x[0,:]):
            unique, cnts = np.unique(y, return_counts=True)
            cnts = dict(zip(unique, cnts))
            root.addChild("leaf", cnts[True] > cnts[False])
            return 
        
        min_gini = 1
        min_feature = None
        min_feature_val = None
        for i in range(x.shape[-1]):
            gini, feature_val = findMinGini(x[:, i], y)
            if detailed:
                print(gini, feature_val, i)
            if gini < min_gini:
                min_gini = gini
                min_feature = i
                min_feature_val = feature_val
        if abs(min_gini - min_gini_old) < 1e-6: # all feature are random
            unique, cnts = np.unique(y, return_counts=True)
            cnts = dict(zip(unique, cnts))
            root.addChild("leaf", cnts[True] > cnts[False])
            return
            
        child_true = root.addChild((min_feature, min_feature_val,), True)
        self.CARTgenerate(x[x[:, min_feature]==min_feature_val], y[x[:, min_feature]==min_feature_val], child_true, detailed, min_gini)
        child_false = root.addChild((min_feature, min_feature_val,), False)
        self.CARTgenerate(x[x[:, min_feature]!=min_feature_val], y[x[:, min_feature]!=min_feature_val], child_false, detailed, min_gini)
    
    # TODO: find nice regularization function
    def pruning(self, root):
        pass
    
    def predict(self, x):
        assert(len(self.root.children) > 0)
        if len(x.shape) == 1:
            tmp = self.root
            if self.type == 'CART':
                while len(tmp.children) > 1:
                    feature = tmp.children[0].key[0]
                    if x[feature] == tmp.children[0].key[1]:
                        tmp = tmp.children[0]
                    else:
                        tmp = tmp.children[1]
                if len(tmp.children) == 1 and tmp.children[0].key == 'leaf':
                    return tmp.children[0].val
            else:
                while len(tmp.children) > 1:
                    feature = tmp.children[0].key
                    if x[feature] == tmp.children[0].val:
                        tmp = tmp.children[0]
                    else:
                        tmp = tmp.children[1]
                if len(tmp.children) == 1 and tmp.children[0].key == 'leaf':
                    return tmp.children[0].val
        else:
            assert(len(x.shape) == 2)
            ans = []
            for test in x:
                ans.append(self.predict(test))
            return ans
            
if __name__ == "__main__":
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

    dt = DecisionTree()
    dt.fit(x, y, type="CART", detailed=True)
    print(dt.root)
    print(dt.predict(x))