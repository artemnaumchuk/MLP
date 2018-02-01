import numpy as np

np.random.seed(1)

# input data
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output data
Y = np.array([[0],
              [1],
              [1],
              [0]])


# logistic (Sigmoid) function
def nonlinear(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


# weights coefficients
syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

# trainings
for j in range(60000):

    # input layer
    l0 = X

    # hidden layer use Sigmoid as activation function on dot product
    # of input data and randomly generated synapse weights
    l1 = nonlinear(np.dot(l0, syn0))

    # output layer
    l2 = nonlinear(np.dot(l1, syn1))

    l2_error = Y - l2

    if (j % 10000) == 0:
        print('Error: ' + str(np.mean(np.abs(l2_error))))

    # back propagate for weight coefficients updates
    # using gradient descent to find local minima of error function
    l2_delta = l2_error * nonlinear(l2, derivative=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlinear(l1, derivative=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

# output
print('Predicted value:')
print(l2)
