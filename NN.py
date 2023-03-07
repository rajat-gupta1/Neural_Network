import numpy as np
import sys
import random
import math
from mnist import MNIST
import time
# from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_dash(x):
    sigx = (1 / (1 + np.exp(-x)))
    return sigx * (1 - sigx)

def softplus(x):
    return np.log(1 + np.exp(x))

def softplus_dash(x):
    return np.exp(x) / (1 + np.exp(x))

def b_w_initialisation(nh, nl, input_nodes, output_nodes):
    b_list = []
    b_list.append(np.zeros((nh, 1)))
    for i in range(nl - 1):
        b_list.append(np.zeros((nh, 1)))
    b_list.append(np.zeros((output_nodes, 1)))

    w_list = []
    w_list.append(np.random.normal(0, 1, size=(nh, input_nodes)))
    # w_list.append(np.ones((nh, input_nodes)))
    for i in range(nl - 1):
        w_list.append(np.random.normal(0, 1, size=(nh, nh)))
        # w_list.append(np.ones((nh, nh)))
    w_list.append(np.random.normal(0, 1, size=(output_nodes, nh)))
    # w_list.append(np.ones((output_nodes, nh)))
    return b_list, w_list

def Forward_helper(a, z, iter, param2, w_list, b_list, cond):
    z.append(np.matmul(w_list[iter], param2) + b_list[iter])
    if cond == 1:
        a.append(sigmoid(z[iter]))
    return a, z

def Forward_prop(x, y, w_list, b_list, nl, Error, i, input_nodes):
    z = []
    a = []
    # Forward
    input = np.array(x[i]).reshape(input_nodes, 1)
    # z.append(np.matmul(w_list[0], input) + b_list[0])
    # a.append(sigmoid(z[0]))
    a, z = Forward_helper(a, z, 0, input, w_list, b_list, 1)

    for k in range(nl - 1):
        a, z = Forward_helper(a, z, k + 1, a[-1], w_list, b_list, 1)

    a, z = Forward_helper(a, z, -1, a[-1], w_list, b_list, 0)
    yhat = sigmoid(z[-1])
    if y[i] == 0:
        y_i = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(10, 1)
    elif y[i] == 1:
        y_i = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(10, 1)
    elif y[i] == 2:
        y_i = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).reshape(10, 1)
    elif y[i] == 3:
        y_i = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).reshape(10, 1)
    elif y[i] == 4:
        y_i = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).reshape(10, 1)
    elif y[i] == 5:
        y_i = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).reshape(10, 1)
    elif y[i] == 6:
        y_i = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).reshape(10, 1)
    elif y[i] == 7:
        y_i = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).reshape(10, 1)
    elif y[i] == 8:
        y_i = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).reshape(10, 1)
    else:
        y_i = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(10, 1)
    this_error = np.linalg.norm(y_i - yhat)
    Error += (this_error) ** 2
    del_error = yhat - y_i
    return Error, a, z, del_error, input

def Backward_prop(der_blist, der_wlist, del_error, a, z, w_list, input, nl, output_nodes):
    del_l2 = del_error.reshape(output_nodes, 1) * sigmoid_dash(z[-1])
    der_blist[-1] += del_l2 
    der_wlist[-1]  += np.matmul(del_l2, a[-1].T)

    for k in range(nl - 1):
        del_l1 = np.matmul(w_list[-1 - k].T, del_l2) * sigmoid_dash(z[-2 - k])
        der_blist[-2 - k] += del_l1
        der_wlist[-2 - k] += np.matmul(del_l1, a[-2 - k].T)
        del_l2 = del_l1
    
    del_l1 = np.matmul(w_list[1].T, del_l2) * sigmoid_dash(z[0])
    der_blist[0] += del_l1
    der_wlist[0] += np.matmul(del_l1, input.T)
    return der_blist, der_wlist

def main():

    mndata = MNIST('data')
    X, y = mndata.load_training()
    X = np.array(X)
    y = np.array(y)
    X = X / 255

    (train_data, test_data, train_labels, test_labels) = train_test_split(X, y, test_size=0.9)

    X = train_data
    y = train_labels

    alpha = 0.1
    input_size = len(train_data)
    nl = int(sys.argv[1])
    nh = int(sys.argv[2])
    ne = int(sys.argv[3])
    nb = int(sys.argv[4])
    output_nodes = 10
    input_nodes = 784
    

    tic = time.perf_counter()

    b_list, w_list = b_w_initialisation(nh, nl, input_nodes, output_nodes)

    for j in range(ne):
        selection_list = np.random.choice(input_size, input_size, replace=False)
        loop_cnt = 0
        cum_error = 0
        for l in range(int(input_size / nb)):
            # Error = np.zeros((output_nodes, 1))
            Error = 0
            del_error = np.zeros((output_nodes, 1))
            der_blist = [0 for _ in range(nl + 1)]
            der_wlist = [0 for _ in range(nl + 1)]

            for i in (selection_list[loop_cnt * nb: (loop_cnt + 1) * nb]):
                Error, a, z, del_error, input = Forward_prop(X, y, w_list, b_list, nl, Error, i, input_nodes)
                der_blist, der_wlist = Backward_prop(der_blist, der_wlist, del_error, a, z, w_list, input, nl, output_nodes)

            Error = Error / nb / 2 
            
            for k in range(nl + 1):
                w_list[k] -= alpha * der_wlist[k] / nb
                b_list[k] -= alpha * der_blist[k] / nb
            
            loop_cnt += 1
            cum_error += Error
        
        cum_error /= loop_cnt
        if j % 50 == 0:
            print(cum_error)

        if cum_error <= 0.01:
            break
    
    print(Error)

    toc = time.perf_counter()
    print(f"Time = {toc - tic:0.4f} seconds")

    correct = 0

    for i in range(input_size):
        z = []
        a = []
        input = np.array(X[i]).reshape(input_nodes, 1)
        a, z = Forward_helper(a, z, 0, input, w_list, b_list, 1)

        for k in range(nl - 1):
            a, z = Forward_helper(a, z, k + 1, a[-1], w_list, b_list, 1)

        a, z = Forward_helper(a, z, -1, a[-1], w_list, b_list, 0)
        yhat = sigmoid(z[-1])
        if np.argmax(yhat) == y[i]:
            correct += 1
    
    print(f"Train Accuracy = {(correct / input_size)}")

    # correct = 0

    # for i in range(input_size, 60000):
    #     z = []
    #     a = []
    #     input = np.array(X[i]).reshape(input_nodes, 1)
    #     a, z = Forward_helper(a, z, 0, input, w_list, b_list, 1)

    #     for k in range(nl - 1):
    #         a, z =  Forward_helper(a, z, k + 1, a[-1], w_list, b_list, 1)

    #     a, z = Forward_helper(a, z, -1, a[-1], w_list, b_list, 0)
    #     yhat = sigmoid(z[-1])
    #     if np.argmax(yhat) == y[i]:
    #         correct += 1
    
    # print(f"Test Accuracy = {correct / (60000 - input_size)}")

    correct = 0

    Xtest, ytest = mndata.load_testing()
    Xtest = np.array(Xtest)
    ytest = np.array(ytest)
    Xtest = Xtest / 255

    for i in range(10000):
        z = []
        a = []
        input = np.array(Xtest[i]).reshape(input_nodes, 1)
        a, z = Forward_helper(a, z, 0, input, w_list, b_list, 1)

        for k in range(nl - 1):
            a, z =  Forward_helper(a, z, k + 1, a[-1], w_list, b_list, 1)

        a, z = Forward_helper(a, z, -1, a[-1], w_list, b_list, 0)
        yhat = sigmoid(z[-1])
        if np.argmax(yhat) == ytest[i]:
            correct += 1
    
    print(f"Test Accuracy 2 = {correct / 10000}")

if __name__ == "__main__":
    main()