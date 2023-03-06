from random import random
n = input("Enter n, the number of points: ")
alpha = input("Enter α, learning rate: ")
epsilon = input("Enter ϵ, the convergence criteria: ")

y = []
x = []

# n = 100
# alpha = 0.0002
# epsilon = 10e-6
n = int(n)
alpha = float(alpha)
epsilon = float(epsilon)

for i in range(n):
    x.append(i)
    y.append(x[i] * 3 + 10 + random() * 2 - 1)

temp = random()
if temp < 0.5:
    m = random()
else:
    m = 1 / random()
b = 2 * random() - 1

k = 0
while(1):
    del_m = 0
    del_b = 0
    for i in range(n):
        del_m += (2 * b * x[i] + 2 * m * (x[i] ** 2) - 2 * x[i] * y[i])
        del_b += (2 * b + 2 * m * x[i] - 2 * y[i])

    del_m /= n
    del_b /= n

    m = m - alpha * del_m
    b = b - alpha * del_b

    if abs(del_m) < epsilon and abs(del_b) < epsilon:
        print(f"Optimal parameters are m={m} and b={b}")
        print(f"Number of iterations: {k}")
        break
    k += 1