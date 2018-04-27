import numpy as np
from perceptron_primary import Perceptron

p = Perceptron()
p.fit(np.array([[3, 3], 
                [4, 3],
                [1, 1]]), np.array([1, 1, -1]), detailed=True)
print(p.predict(np.array([[3, 3], 
                          [4, 3],
                          [1, 1]])))
                          
