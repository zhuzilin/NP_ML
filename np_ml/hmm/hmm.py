import numpy as np

class HMM:
    def __init__(self, A=None, B=None, pi=None):
        self.A = A
        self.B = B
        self.pi = pi
    def forward(self, O, detailed=False):
        alpha = self.pi*self.B[:, O[0]]
        if detailed:
            print("alpha: {}".format(alpha))
        for t in range(1, len(O)):
            alpha = np.squeeze(np.matmul(self.A.T, alpha[..., None]))*self.B[:, O[t]]
        return np.sum(alpha)
        
    def backward(self, O):
        beta = np.ones(self.pi.shape)
        for t in range(len(O)-1, 0, -1):
            beta = np.squeeze(np.matmul(self.A, (self.B[:, O[t]]*beta)[..., None]))
        return np.sum(self.pi*self.B[:, O[0]]*beta)
        
    def Viterbi(self, O):
        # initialization
        delta = self.pi*self.B[:, O[0]]
        psi = np.zeros((len(O), len(delta))) # t*i
        for t in range(1, len(O)):
            delta = np.max(delta[..., None]*self.A, axis=-1)*self.B[:, O[t]]
            psi[t, :] = np.argmax(delta[..., None]*self.A, axis=-1)
        I = []
        i = np.argmax(delta)
        I.append(i)
        for t in range(len(O)-1, 0, -1):
            i = psi[t, int(i)]
            I.append(i)
        I = [int(i) for i in I]
        return np.max(delta), I[::-1]