from units import Perceptron
import math

def threshold(x):
    return 0 if x < 0 else 1

input_vals = [[0, 0], [0, 1], [1, 0], [1, 1]]
for i in input_vals:
    in1 = Perceptron([1, i[0], i[1]], [-1, -1, 1], activation_func=threshold)
    in2 = Perceptron([1, i[0], i[1]], [-1, 1, -1], activation_func=threshold)
    h1 = Perceptron([1, in1, in2], [-1, 1, 1], activation_func=threshold)
    print(i, h1.solve())
