import numpy as np

# a perceptron is defined as a single unit
class Perceptron:
    # bias can be a function
    # eta is learning rate, used for backpropagation
    def __init__(self, units, weights, activation_func, bias=None, eta=None, node_out=None, node_in=None):
        intermediate = False
        tmp = []
        for unit in units:
            if type(unit) is Perceptron:
                intermediate = True
                tmp.append(unit.solve())
            else:
                tmp.append(unit)
        if intermediate is True:
            self.units = np.array(tmp)
        else:
            self.units = np.array(units) # x1, x2, etc
        self.weights = np.array(weights)
        self.activation_func = activation_func
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

    def solve(self):
        return self.activation_func(self.net())
