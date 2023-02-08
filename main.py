
from tqdm import trange
import pickle
from keras.datasets import mnist
import numpy as np
np.set_printoptions(suppress=True)


(raw_X_train, raw_y_train), (raw_X_test, raw_y_test) = mnist.load_data()
X_train, y_train, X_test, y_test = [], [], [], []

# Processing data
for i in range(len(raw_X_train)):
    X_train.append(raw_X_train[i].flatten()/255.0)
for i in range(len(raw_X_test)):
    X_test.append(raw_X_test[i].flatten()/255.0)

num_classes = 10

for i in range(len(raw_y_train)):
    y_train.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_train[i][raw_y_train[i]] = 1.
    y_train[i] = np.array(y_train[i])

for i in range(len(raw_y_test)):
    y_test.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_test[i][raw_y_test[i]] = 1.
    y_test[i] = np.array(y_test[i])

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

input_node_size = 784
hidden_node_size = 128
output_node_size = 10


def tanh(x, deriv=False):
    if deriv:
        return 1.0 - (np.power(x, 2))
    return np.tanh(x)


def relu(x, deriv=False):
    if deriv:
        x[x < 0] = 0
        x[x > 0] = 1
        return x
    return np.maximum(0, x)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


num_epochs = 10000

W1 = np.random.randn(input_node_size, hidden_node_size)
W2 = np.random.randn(hidden_node_size, output_node_size)
BS = 2500
lr = 1e-3


for epoch in (t := trange(num_epochs)):
    samp = np.random.randint(0, X_train.shape[0], size=(BS))
    X = X_train[samp]
    Y = y_train[samp]

    z1 = X.dot(W1)
    a1 = tanh(z1)
    z2 = a1.dot(W2)
    a2 = softmax(z2)

    loss = -np.mean(np.sum(Y * np.log(a2+1e-20), axis=1))
    t.set_description("Loss: " + str(np.round(loss, 5)))

    dz2 = a2 - Y
    dw2 = a1.T.dot(dz2)
    dz1 = dz2.dot(W2.T) * (1. - np.power(a1, 2))
    dw1 = X.T.dot(dz1)

    W1 -= lr * dw1
    W2 -= lr * dw2

with open("best_loss.bin", "rb") as f:
    prev_loss = pickle.load(f)
if (np.round(loss, 5) <= prev_loss):

    with open("W1.bin", "wb") as g:
        pickle.dump(W1, g)

    with open("W2.bin", "wb") as h:
        pickle.dump(W2, h)
    with open("best_loss.bin", "wb") as i:
        pickle.dump(loss, i)
    print("New best weights found.")
