import numpy as np
import sys
import random
import math
from mnist import MNIST
import time
from sklearn.model_selection import train_test_split

def sigmoid(x):
    # Function to find the sigmoid function
    return 1 / (1 + np.exp(-x))

def sigmoid_dash(x):
    # Function to find the derivative of sigmoid function
    sigx = (1 / (1 + np.exp(-x)))
    return sigx * (1 - sigx)

def b_w_initialisation(nh, nl, input_nodes, output_nodes):
    # Function to initialise b with 0s and w with random numbers
    b_list = []
    b_list.append(np.zeros((nh, 1)))
    for i in range(nl - 1):
        b_list.append(np.zeros((nh, 1)))
    b_list.append(np.zeros((output_nodes, 1)))

    w_list = []
    w_list.append(np.random.normal(0, 1, size=(nh, input_nodes)))
    for i in range(nl - 1):
        w_list.append(np.random.normal(0, 1, size=(nh, nh)))
    w_list.append(np.random.normal(0, 1, size=(output_nodes, nh)))
    return b_list, w_list

def Forward_helper(a, z, iter, param2, w_list, b_list, cond):
    # Function to help forward propagation
    z.append(np.matmul(w_list[iter], param2) + b_list[iter])
    if cond == 1:
        a.append(sigmoid(z[iter]))
    return a, z

def Forward_prop(x, y, w_list, b_list, nl, Error, i, input_nodes):
    # Function to do forward propagation
    z = []
    a = []
    # Forward
    input = np.array(x[i]).reshape(input_nodes, 1)
    a, z = Forward_helper(a, z, 0, input, w_list, b_list, 1)

    # For more than one hidden layers
    for k in range(nl - 1):
        a, z = Forward_helper(a, z, k + 1, a[-1], w_list, b_list, 1)

    a, z = Forward_helper(a, z, -1, a[-1], w_list, b_list, 0)

    # Creating an array for y_i similar to the one given in the train data
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

    # Finding delta of error
    del_error = yhat - y_i
    return Error, a, z, del_error, input

def Backward_prop(der_blist, der_wlist, del_error, a, z, w_list, input, nl, output_nodes):
    # Function to do backward propagation
    del_l2 = del_error.reshape(output_nodes, 1) * sigmoid_dash(z[-1])
    der_blist[-1] += del_l2
    der_wlist[-1]  += np.matmul(del_l2, a[-1].T)

    # For more than one hidden layers
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

    # Using MNIST data
    mndata = MNIST('data')
    X, y = mndata.load_training()
    X = np.array(X)
    y = np.array(y)

    # Normalising data
    X = X / 255

    # Splitting data into train and test
    (train_data, test_data, train_labels, test_labels) = train_test_split(X, y, test_size=70)

    X = train_data
    y = train_labels

    # The learning rate
    alpha = 0.1
    input_size = len(train_data)

    # Number of hidden layers
    nl = int(sys.argv[1])

    # Number of nodes in each hidden layer
    nh = int(sys.argv[2])

    # Number of epochs
    ne = int(sys.argv[3])

    # Batch size
    nb = int(sys.argv[4])

    # Size of input and output nodes
    output_nodes = 10
    input_nodes = 784
    
    tic = time.perf_counter()

    # Initializing b and w
    b_list, w_list = b_w_initialisation(nh, nl, input_nodes, output_nodes)
    prev_cum_error = 0
    for j in range(ne):
        selection_list = np.random.choice(input_size, input_size, replace=False)
        loop_cnt = 0
        cum_error = 0

        # Iterating over each epoch
        for l in range(int(input_size / nb)):
            Error = 0
            del_error = np.zeros((output_nodes, 1))
            der_blist = [0 for _ in range(nl + 1)]
            der_wlist = [0 for _ in range(nl + 1)]

            for i in (selection_list[loop_cnt * nb: (loop_cnt + 1) * nb]):
                Error, a, z, del_error, input = Forward_prop(X, y, w_list, b_list, nl, Error, i, input_nodes)
                der_blist, der_wlist = Backward_prop(der_blist, der_wlist, del_error, a, z, w_list, input, nl, output_nodes)

            # Finding error
            Error = Error / nb / 2 
            
            # Updating w and b for the next loop
            for k in range(nl + 1):
                w_list[k] -= alpha * der_wlist[k] / nb
                b_list[k] -= alpha * der_blist[k] / nb
            
            loop_cnt += 1
            cum_error += Error
        
        # Calculating and printing error
        cum_error /= loop_cnt
        print(f"Iteration: {j}")
        print(f"Error: {cum_error}")

        # Break condition
        if cum_error <= 0.03 or abs(cum_error - prev_cum_error) < 10e-6:
            break
        prev_cum_error = cum_error

    toc = time.perf_counter()
    print(f"Time = {toc - tic:0.4f} seconds")

    # Finding train and test accuracies
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

    correct = 0

    for i in range(len(test_data)):
        z = []
        a = []
        input = np.array(test_data[i]).reshape(input_nodes, 1)
        a, z = Forward_helper(a, z, 0, input, w_list, b_list, 1)

        for k in range(nl - 1):
            a, z =  Forward_helper(a, z, k + 1, a[-1], w_list, b_list, 1)

        a, z = Forward_helper(a, z, -1, a[-1], w_list, b_list, 0)
        yhat = sigmoid(z[-1])
        if np.argmax(yhat) == test_labels[i]:
            correct += 1
    
    print(f"Test Accuracy = {correct / len(test_data)}")

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