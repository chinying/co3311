from units import Perceptron

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

def threshold(net):
    return 0 if net < 0 else 1

# AND gate config
def and_bias(u):
    return -1 * len(u.units)

for i in inputs:
    or_gate = Perceptron([1, i[0], i[1]], [-1, 1, 1], activation_func=threshold)
    and_gate = Perceptron([i[0], i[1]], [1, 1], bias=and_bias, activation_func=threshold)
    print("values of inputs " + str(i))
    print("OR %d, AND %d" % (or_gate.solve(), and_gate.solve()))

