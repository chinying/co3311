from units import Perceptron

p = Perceptron([2, 2, 4, -22, -3], [1, -1], [])

def nn_func(a):
    return filter(lambda x: x < 1, a)

# print(p.solve(nn_func))
# print(list(nn_and(np.array([15, -2, 3]))))

# OR gate
def or_net(w, x):
    return sum(w * x)
def or_bias(u, b, w):
    return b[0] * b[1]
def or_output(net):
    if net < 0:
        return 0
    else:
        return 1
print("OR gate modelled with perceptron")
or_gate = Perceptron([0, 0], [1, -1], [1, 1])
print(or_gate.solve(or_output, or_net, or_bias))

# AND gate
def and_net(w, x):
    return sum(w * x)
def and_bias(u, b, w):
    return -1 * len(u) 
def and_output(net):
    if net < 0:
        return 0
    else:
        return 1
print("AND gate modelled with perceptron")
and_gate = Perceptron([1, 1], [1, 0], [1, 1])
print(and_gate.solve(and_output, and_net, and_bias))
