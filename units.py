import numpy as np

# a perceptron is defined as a single unit
class Perceptron:
    # bias can be a function
    # bias taken in as a list[bias, wt]
    def __init__(self, units, bias, weights):
        self.units = np.array(units) # x1, x2, etc
        self.bias = bias
        self.weights = np.array(weights)

    def net(self, func, bias_func):
        bias = bias_func(self.units, self.bias, self.weights) 
        return bias + func(self.units, self.weights) 

    def solve(self, func, net_func, bias_func):
        return func(self.net(net_func, bias_func))
