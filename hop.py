from units import Perceptron
import numpy as np
from copy import deepcopy

# shall try with 3 units first
class Hopfield:
    def __init__(self, wt, size=3):
        # there are size + 1 units inc bias
        self.size = size
        tmp = []
        for i in range(1, size + 1, 1):
            tmp.append([0] * i + wt[i-1])
        tmp.append([0 for i in range(size + 1)])
        self.weights = np.array(tmp)
        self.weights += self.weights.T # symmetric matrix
        print(self.weights)
        # self.u1 = Perceptron([1,0,0,0], self.weights[0], activation_func=self.threshold)
        # print(self.u1.solve())
        self.units = []
        states = self.states()
        for cnt, state in enumerate(states):
            # Perceptron([1] + state)
            after = []
            units = ([1] + state[::-1]) # because of big endian
            for b in range(size, 0, -1): # big-endian
                # print(units, self.weights[b])
                p = Perceptron(units, self.weights[b], activation_func=self.threshold)
                after.append(p.solve())
            # print(state, after)
            # this checks for stable state, True = stable
            print(cnt, [cnt] * self.size == self.trans_state(state, after))

    # if you care about performance, divide 2 each iteration and then concat lists
    def bits(self, n, maxlen):
        tmp = str(bin(n)[2:]).zfill(maxlen)
        return [int(t) for t in tmp]

    def threshold(self, x): # change threshold if required
        return 1 if x > 0 else 0

    # compare nth bit, copy the rest
    def trans_state(self, b1, b2):
        # all 3 sets will point to same item (ie. change together) if you don't deepcopy
        b3 = [deepcopy(b1) for i in range(self.size)]
        for i in range(self.size):
            if b1[i] != b2[i]:
                b3[i][i] = b2[i]
        return [int("".join(str(s) for s in num), 2) for num in b3]

    def states(self):
        s = []
        n = 2 ** self.size
        for i in range(n):
            s.append(self.bits(i, self.size))
        return s

# weights = [[-.5, .2, .75], [.3, -.4], [-.8]]
weights = [[.2, -.4, -.1], [.3, -.3], [.2]]
h = Hopfield(weights, 3)
# print(h.states())
