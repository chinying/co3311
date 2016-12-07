from units import Perceptron
import math

def sigmoid(x):
    return 1 / (1 + math.e**(-x))

def activation(net):
    return sigmoid(net)

def epsilon(t, a):
    return (t - a) * a * (1 - a)

def delta_wh(eta, epsilon, a):
    return eta * epsilon * a

# init values
learning = 0.6
_w = [0.1, 0.4, -.2, 0.2, 0.2, -0.5]

# pass 1
x1, x2 = 0.4, -0.7
h1 = Perceptron([x1, x2], [_w[0], _w[2]])
h2 = Perceptron([x1, x2], [_w[1], _w[3]])
y = Perceptron([h1.solve(sigmoid), h2.solve(sigmoid)], [_w[4], _w[5]])
error31 = epsilon(0.1, y.solve(sigmoid))
dwh21 = (delta_wh(learning, error31, h1.solve(sigmoid)))
dwh22 = (delta_wh(learning, error31, h2.solve(sigmoid)))
oldwh1, _w[4] = _w[4], _w[4] + dwh21
oldwh2, _w[5] = _w[5], _w[5] + dwh22

error21 = error31 * h1.solve(sigmoid) * (1 - h1.solve(sigmoid)) * oldwh1
error22 = error31 * h2.solve(sigmoid) * (1 - h2.solve(sigmoid)) * oldwh2

dw1h1 = (delta_wh(learning, error21, x1))
dw1h2 = (delta_wh(learning, error22, x1))
dw2h1 = (delta_wh(learning, error21, x2))
dw2h2 = (delta_wh(learning, error22, x2))

_w[0] = _w[0] + dw1h1
_w[1] = _w[1] + dw1h2
_w[2] = _w[2] + dw2h1
_w[3] = _w[3] + dw2h2
print(dw1h1, dw1h2, dw2h1, dw2h2)
print(_w[:4])

# pass 2
x1, x2 = 0.3, -0.5
h1 = Perceptron([x1, x2], [_w[0], _w[2]])
h2 = Perceptron([x1, x2], [_w[1], _w[3]])
y = Perceptron([h1.solve(sigmoid), h2.solve(sigmoid)], [_w[4], _w[5]])
error31 = epsilon(0.1, y.solve(sigmoid))
dwh21 = (delta_wh(learning, error31, h1.solve(sigmoid)))
dwh22 = (delta_wh(learning, error31, h2.solve(sigmoid)))
oldwh1, _w[4] = _w[4], _w[4] + dwh21
oldwh2, _w[5] = _w[5], _w[5] + dwh22

error21 = error31 * h1.solve(sigmoid) * (1 - h1.solve(sigmoid)) * oldwh1
error22 = error31 * h2.solve(sigmoid) * (1 - h2.solve(sigmoid)) * oldwh2

dw1h1 = (delta_wh(learning, error21, x1))
dw1h2 = (delta_wh(learning, error22, x1))
dw2h1 = (delta_wh(learning, error21, x2))
dw2h2 = (delta_wh(learning, error22, x2))

_w[0] = _w[0] + dw1h1
_w[1] = _w[1] + dw1h2
_w[2] = _w[2] + dw2h1
_w[3] = _w[3] + dw2h2
print(dw1h1, dw1h2, dw2h1, dw2h2)
print(_w[:4])

# pass 3
x1, x2 = 0.6, 0.1
h1 = Perceptron([x1, x2], [_w[0], _w[2]])
h2 = Perceptron([x1, x2], [_w[1], _w[3]])
y = Perceptron([h1.solve(sigmoid), h2.solve(sigmoid)], [_w[4], _w[5]])
error31 = epsilon(0.1, y.solve(sigmoid))
dwh21 = (delta_wh(learning, error31, h1.solve(sigmoid)))
dwh22 = (delta_wh(learning, error31, h2.solve(sigmoid)))
oldwh1, _w[4] = _w[4], _w[4] + dwh21
oldwh2, _w[5] = _w[5], _w[5] + dwh22

error21 = error31 * h1.solve(sigmoid) * (1 - h1.solve(sigmoid)) * oldwh1
error22 = error31 * h2.solve(sigmoid) * (1 - h2.solve(sigmoid)) * oldwh2

dw1h1 = (delta_wh(learning, error21, x1))
dw1h2 = (delta_wh(learning, error22, x1))
dw2h1 = (delta_wh(learning, error21, x2))
dw2h2 = (delta_wh(learning, error22, x2))

_w[0] = _w[0] + dw1h1
_w[1] = _w[1] + dw1h2
_w[2] = _w[2] + dw2h1
_w[3] = _w[3] + dw2h2
print(dw1h1, dw1h2, dw2h1, dw2h2)
print(_w[:4])

# pass 4
x1, x2 = 0.2, 0.4
h1 = Perceptron([x1, x2], [_w[0], _w[2]])
h2 = Perceptron([x1, x2], [_w[1], _w[3]])
y = Perceptron([h1.solve(sigmoid), h2.solve(sigmoid)], [_w[4], _w[5]])
error31 = epsilon(0.1, y.solve(sigmoid))
dwh21 = (delta_wh(learning, error31, h1.solve(sigmoid)))
dwh22 = (delta_wh(learning, error31, h2.solve(sigmoid)))
oldwh1, _w[4] = _w[4], _w[4] + dwh21
oldwh2, _w[5] = _w[5], _w[5] + dwh22

error21 = error31 * h1.solve(sigmoid) * (1 - h1.solve(sigmoid)) * oldwh1
error22 = error31 * h2.solve(sigmoid) * (1 - h2.solve(sigmoid)) * oldwh2

dw1h1 = (delta_wh(learning, error21, x1))
dw1h2 = (delta_wh(learning, error22, x1))
dw2h1 = (delta_wh(learning, error21, x2))
dw2h2 = (delta_wh(learning, error22, x2))

_w[0] = _w[0] + dw1h1
_w[1] = _w[1] + dw1h2
_w[2] = _w[2] + dw2h1
_w[3] = _w[3] + dw2h2
print(dw1h1, dw1h2, dw2h1, dw2h2)
print(_w[:4])

# pass 5
x1, x2 = 0.1, -0.2
h1 = Perceptron([x1, x2], [_w[0], _w[2]])
h2 = Perceptron([x1, x2], [_w[1], _w[3]])
y = Perceptron([h1.solve(sigmoid), h2.solve(sigmoid)], [_w[4], _w[5]])
error31 = epsilon(0.1, y.solve(sigmoid))
dwh21 = (delta_wh(learning, error31, h1.solve(sigmoid)))
dwh22 = (delta_wh(learning, error31, h2.solve(sigmoid)))
oldwh1, _w[4] = _w[4], _w[4] + dwh21
oldwh2, _w[5] = _w[5], _w[5] + dwh22

error21 = error31 * h1.solve(sigmoid) * (1 - h1.solve(sigmoid)) * oldwh1
error22 = error31 * h2.solve(sigmoid) * (1 - h2.solve(sigmoid)) * oldwh2

dw1h1 = (delta_wh(learning, error21, x1))
dw1h2 = (delta_wh(learning, error22, x1))
dw2h1 = (delta_wh(learning, error21, x2))
dw2h2 = (delta_wh(learning, error22, x2))

_w[0] = _w[0] + dw1h1
_w[1] = _w[1] + dw1h2
_w[2] = _w[2] + dw2h1
_w[3] = _w[3] + dw2h2
print(dw1h1, dw1h2, dw2h1, dw2h2)
print(_w[:4])

