import numpy as np

# a perceptron is defined as a single unit
class Perceptron:
    # bias can be a function
    def __init__(self, units, bias, weights):
        self.units = np.array(units) # x1, x2, etc
        self.bias = bias
        self.weights = np.array(weights)

    def net(self):
        return self.bias + m # m is defined as no. inputs > 1

    def solve(self, func):
        # do something with the units
        return list(func(self.units))

p = Perceptron([2, 2, 4, -22, -3], 5, [])

def nn_func(a):
    return filter(lambda x: x < 1, a)

# print(p.solve(nn_func))
# print(list(nn_and(np.array([15, -2, 3]))))
