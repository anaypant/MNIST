import pandas as pd
import numpy as np
from ast import literal_eval
from tqdm import trange
import pickle
from keras.datasets import mnist

data_name = input("Which dataset would you like to train? (Excluding .csv) ")
print("Preproccessing csv ... ")
if data_name == "MNIST":
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
else:
    df = pd.read_csv("datasets/"+str(data_name)+".csv")
    classes = (df["label"].unique()).tolist()
    num_classes = len(df["label"].unique())
    raw_x = df["data"].apply(literal_eval)
    raw_y = df["label"].to_list()
    X_train = []
    y_train = []

    for x in raw_x:
        X_train.append(np.array(x))
    X_train = np.array(X_train)
    for y in range(len(raw_y)):
        y_train.append([])
        for _ in range(num_classes):
            y_train[y].append(0)
        y_train[y][classes.index(raw_y[y])]=1
        
    y_train = np.array(y_train)
print("Processed.")

input_node_size = 784
hidden_node_size = 128
output_node_size = num_classes


def tanh(x, deriv=False):
    if deriv:
        return 1.0 - (np.power(x, 2))
    return np.tanh(x)


# def relu(x, deriv=False):
#     if deriv:
#         x[x < 0] = 0
#         x[x > 0] = 1
#         return x
#     return np.maximum(0, x)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


num_epochs = 10000

W1 = np.random.randn(input_node_size, hidden_node_size)
W2 = np.random.randn(hidden_node_size, output_node_size)
BS = min(len(X_train), 128)
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
    dz1 = dz2.dot(W2.T) * tanh(a1,deriv=True)#(1. - np.power(a1, 2))
    dw1 = X.T.dot(dz1)

    W1 -= lr * dw1
    W2 -= lr * dw2



with open("weights/W1_"+str(data_name)+".bin", "wb") as f:
    pickle.dump(W1, f)
with open("weights/W2_"+str(data_name)+".bin", "wb") as g:
    pickle.dump(W2, g)
with open("losses/" + data_name+"_best_loss.bin", "wb") as h:
    pickle.dump(np.round(loss,5), h)
print("Training completed.")