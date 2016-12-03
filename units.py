import numpy as np

# a perceptron is defined as a single unit
class Perceptron:
    # bias can be a function
    def __init__(self, units, weights, bias=None):
        self.units = np.array(units) # x1, x2, etc
        self.weights = np.array(weights)
        if bias == None:
            self.bias = (self.units[0], self.weights[0])
        else:
            self.bias_func = bias

    # net is just a linear combination of units with weights
    def net(self):
        try:
            return self.bias_func(self) + sum(self.units * self.weights)
        except AttributeError:
            return sum(self.units * self.weights)

    def solve(self, func):
        return func(self.net())
