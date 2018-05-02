import numpy as np
from np_ml import HMM

if __name__ == '__main__':
    print("--------------------------------------------------------")
    print("Hidden Markov Model simple example!")
    print("example in Statistical Learning Method(《统计学习方法》)")
    print("--------------------------------------------------------")
    # calculate probability
    A = np.array([[0.5, 0.2, 0.3],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5]])
    B = np.array([[0.5, 0.5],
                  [0.4, 0.6],
                  [0.7, 0.3]])
    pi = np.array([0.2, 0.4, 0.4])
    
    hmm = HMM(A, B, pi)

    O = np.array([0, 1, 0])
    print("A: ")
    print(A)
    print("B: ")
    print(B)
    print("pi: ")
    print(pi)
    print("O: ")
    print(O)
    print("")
    
    print("Forward: ")
    print(hmm.forward(O))
    print("Backward: ")
    print(hmm.backward(O))

    # learning
    # We will not implement the supervised version, since it need huge amount of data
    # Baum-Welch algorithm, which is an application of EM

    # predict
    # Viterbi
    print("Viterbi: ")
    max_prob, path = hmm.Viterbi(O)
    print("Maximum probability is:", max_prob, " and path is", path)