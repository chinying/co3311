from units import Perceptron
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time

def threshold(x):
    return 0 if x < 0 else 1

def sigmoid(x):
    return 1 / (1 + math.e**(-x))

def widrow_hoff(expected):
    corrected = (self.solve() - expected) ** 2
    print(corrected)

# TODO, split the below into categories, constants, variables, tmp / scope

def run(weights, target, learning=0.1, plot=False):
    rounds = 2000
    rounds += 1
    learning = 0.1
    bias = -1
    inputs = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]] # fixed 4 per epoch
    targets = target
    wt = weights # this is fixed at epoch 0
    err = 0
    current_wt, new_wt = np.array(wt), []

    past4 = []

    for i in range(rounds * 4):
        epoch = i // 4
        current_input = inputs[i % 4]
        current_target = targets[i % 4]
        # print(i, wt, current_input, current_target)
        p1 = Perceptron(current_input, wt, activation_func=sigmoid)
        a = p1.solve()
        err = current_target - a
        # print("%f + %f * %f * %f %f" % (wt[1], learning, a, (1-a), err))
        new_wt = np.array(wt)
        wt[0] += (learning * err * a * (1 - a) * current_input[0])
        dw1 = learning * err * a * (1 - a) * current_input[1]
        dw2 = learning * err * a * (1 - a) * current_input[2]
        wt[1] += dw1
        wt[2] += dw2
        dww = sum((current_wt - new_wt) ** 2) / learning ** 2
        past4.append(err)
        current_wt = new_wt

    sum4 = [sum(np.array(past4[i-3:i+1]) ** 2) for i in range(len(past4)) if i > 2]
    print(sum4[4000], sum4[8000]) # 1000th and 2000th respectively

    if plot == True:
        plt.plot(sum4)
        plt.show()

    return(sum4)

def genRand():
    targets = [[random.uniform(.1, .9) for i in range(4)] for j in range(5)]
    weights = [[random.uniform(-1, 1) for i in range(3)] for j in range(5)]
    learning = [.01, .1, .25, .5, 1, 2]
    return(targets, weights, learning)
    # print(weights)

def experiment():
    (i, j, k) = genRand()
    cnt = 1
    for x in i:
        for y in j:
            for z in k:
                print(cnt, x, y, z)
                cnt += 1
                if cnt > 10:
                    return


# genRand()
# run(weights = [-1, 1, 1], target = [0, 1, 0, 1], plot=False)
# experiment()

outfile = "files/" + str(time.time()).replace(".", "") + ".txt"
fo = open(outfile, "w")
fo.write(str(genRand()))
fo.close()
