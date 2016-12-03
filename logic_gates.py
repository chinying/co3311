from units import Perceptron

# OR gate
def or_output(net):
    if net < 0:
        return 0
    else:
        return 1
print("OR gate modelled with perceptron")
or_gate = Perceptron([1, 0, 0], [-1, 1, 1])
print(or_gate.solve(or_output))

# AND gate
def and_bias(u):
    return -1 * len(u.units)
def and_output(net):
    if net < 0:
        return 0
    else:
        return 1
print("AND gate modelled with perceptron")
and_gate = Perceptron([1, 1], [1, 1], and_bias)
print(and_gate.solve(and_output))
