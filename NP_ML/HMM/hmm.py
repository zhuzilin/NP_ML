import numpy as np

class HMM:
    def __init__(self, A=None, B=None, pi=None):
        self.A = A
        self.B = B
        self.pi = pi
    def forward(self, O):
        alpha = pi*B[:, O[0]]
        print(alpha)
        for t in range(1, len(O)):
            alpha = np.squeeze(np.matmul(A.T, alpha[..., None]))*B[:, O[t]]
        return np.sum(alpha)
        
    def backward(self, O):
        beta = np.ones(pi.shape)
        for t in range(len(O)-1, 0, -1):
            beta = np.squeeze(np.matmul(A, (B[:, O[t]]*beta)[..., None]))
        return np.sum(pi*B[:, O[0]]*beta)
        
    def Viterbi(self, O):
        # initialization
        delta = pi*B[:, O[0]]
        psi = np.zeros((len(O), len(delta))) # t*i
        for t in range(1, len(O)):
            delta = np.max(delta[..., None]*A, axis=-1)*B[:, O[t]]
            psi[t, :] = np.argmax(delta[..., None]*A, axis=-1)
        I = []
        i = np.argmax(delta)
        I.append(i)
        for t in range(len(O)-1, 0, -1):
            i = psi[t, int(i)]
            I.append(i)
        I = [int(i) for i in I]
        return np.max(delta), I[::-1]
        
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

print("Forward")
print(hmm.forward(O))
print("Backward")
print(hmm.backward(O))

# learning
# We will not implement the supervised version, since it need huge amount of data
# Baum-Welch algorithm, which is an application of EM

# predict
# Viterbi
print("Viterbi: ")
max_prob, path = hmm.Viterbi(O)
print("Maximum probability is:", max_prob, "path is", path)