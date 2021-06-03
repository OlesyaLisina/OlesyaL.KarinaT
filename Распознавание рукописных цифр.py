import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.datasets import mnist

# loading the database
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# making the arrays one-dimensional and inserting 1 at the beginning
X1_train = X_train.reshape(60000, 1, -1)
X1_train = X1_train.reshape(60000, -1)
X1_train = np.insert(X1_train, 0, 1, axis=1)

X1_test = X_test.reshape(10000, 1, -1)
X1_test = X1_test.reshape(10000, -1)
X1_test = np.insert(X1_test, 0, 1, axis=1)


# function for creating sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# function that sets the decision threshold (decision boundary = 0.5)
def predict(X, params):
    return np.round(sigmoid(X @ params))


# function for evaluating accuracy
def prediction(forecast, y):
    tp = np.logical_and(forecast, y)  # true positive
    fp = np.logical_and(forecast, np.logical_not(y))  # false positive
    tn = np.logical_and(np.logical_not(forecast), np.logical_not(y))  # true negative
    fn = np.logical_and(np.logical_not(forecast), y)  # false negative

    # total amount
    tp = np.sum(tp)
    fp = np.sum(fp)
    tn = np.sum(tn)
    fn = np.sum(fn)

    precision = tp / (tp + fp)  # positive predict value
    recall = tp / (tp + fn)  # true positive rate
    score = 2 * (recall * precision) / (recall + precision)  # prediction

    return precision, recall, score


teta = np.zeros((785, 1))  # theta array
teta_new = np.zeros((785, 1))  # theta array for gradient descent
epsilon = 1e-5
summa = 0
TETA = np.zeros((785, 1))

for a in range(10):
    i = 0
    cost_history = np.zeros((1500))
    # single digit selection
    if a == 0:
        y = np.where(y_train == a, 1, 0)
    else:
        y = np.where(y_train == a, 1, 0)

    y = y[:, np.newaxis]

    cost = -y.T @ (np.log(sigmoid(X1_train @ teta))) - \
           (1 - y).T @ (np.log(1 - sigmoid(X1_train @ teta)))
    cost_old = cost + 5

    print()
    print("Цифра - ", a)
    print()
    while (abs(cost - cost_old) > epsilon):
        cost_old = cost
        # use gradient descent algorithm
        teta_new = teta - (X1_train.T @ (sigmoid(X1_train @ teta) - y)) * (0.5 / 60000)
        teta = np.zeros((785, 1))
        teta = teta + teta_new
        # cost function
        cost = -y.T @ (np.log(sigmoid(X1_train @ teta))) - \
               (1 - y).T @ (np.log(1 - sigmoid(X1_train @ teta)))
        cost = cost / 60000
        cost_history[i] = cost
        i = i + 1
    TETA = np.concatenate((TETA, teta), axis=1)
    # plot of convergence of Cost Function
    cost_history = cost_history[0:i]
    plt.figure()
    sns.set_style('white')
    plt.plot(range(len(cost_history)), cost_history, 'r')
    plt.title("График сходимости функции потерь")
    plt.xlabel("Число итераций")
    plt.ylabel("Значение функции при текущем teta")
    plt.show()

    # using test data
    y1 = np.where(y_test == a, 1, 0)
    y1 = y1[:, np.newaxis]

    # calculating the recognition accuracy
    forecast = predict(X1_test, teta)
    u, w, score_test = prediction(forecast, y1)
    print('score =', score_test, '    precision =', u, '  recall =', w)
    print()
TETA = TETA[:, 1:]
