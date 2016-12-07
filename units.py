import numpy as np

# a perceptron is defined as a single unit
class Perceptron:
    # bias can be a function
    # eta is learning rate, used for backpropagation
    def __init__(self, units, weights, bias=None, eta=None):
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

    # TODO, also note that this is currently one epoch
    # TODO, batched vs unbatched learning
    # TODO, variables currently undefined
    def learn(self, error_func, expected):
        try:
            delta_wh = self.eta * error_func * activation_val
            delta_wih = self.eta * error_func * x
        except AttributeError:
            raise Exception("learning rate not set")

    def solve(self, func):
        return func(self.net())
