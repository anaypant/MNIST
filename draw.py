import pygame
import win32api
import pickle
import numpy as np
import matplotlib.pyplot as plt
import keyboard
from keras.datasets import mnist
import random


# -- boring shit, don't look at this

(raw_X_train, raw_y_train), (raw_X_test, raw_y_test) = mnist.load_data()
X_train, y_train = [], []
non_flattened_X_train = []

# Processing data
for i in range(len(raw_X_train)):
    X_train.append(raw_X_train[i].flatten()/255.0)
    non_flattened_X_train.append(raw_X_train[i]/255.0)

num_classes = 10

for i in range(len(raw_y_train)):
    y_train.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_train[i][raw_y_train[i]] = 1.
    y_train[i] = np.array(y_train[i])

X_train = np.array(X_train)
y_train = np.array(y_train)
non_flattened_X_train = np.array(non_flattened_X_train)
# ---------


# This is more interesting
pygame.init()

with open("W1.bin", "rb") as f:
    W1 = pickle.load(f)
with open("W2.bin", "rb") as g:
    W2 = pickle.load(g)


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
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


samps = [[], [], [], [], [], [], [], [], [], []]
num_samps_per_class = 20
for curr_num in range(10):
    counter = 0
    while (len(samps[curr_num]) != num_samps_per_class):
        if raw_y_train[counter] == curr_num:
            samps[curr_num].append(non_flattened_X_train[counter])
        counter += 1

CELL_W = 10
screen = pygame.display.set_mode((28 * CELL_W, 28 * CELL_W))
grid = []
for i in range(28):
    grid.append([])
    for j in range(28):
        grid[i].append(0)
newGrid = np.array(grid, copy=True).flatten()
clock = pygame.time.Clock()
while True:
    screen.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.MOUSEMOTION and win32api.GetKeyState(0x01) < 0:
            mX, mY = pygame.mouse.get_pos()
            i = mY//CELL_W
            j = mX//CELL_W
            grid[i][j] += random.uniform(0.2, 0.6)
            grid[i][j] = min(1.0, grid[i][j])

            newGrid = np.array(grid, copy=True).flatten()

    if (keyboard.is_pressed("1")):
        # get a random sample of 1 and draw it
        # copy samps[1] to crid
        rand_samp = random.choice(samps[1])
        grid = rand_samp.tolist()
        newGrid = np.array(grid, copy=True).flatten()
    elif (keyboard.is_pressed("2")):
        # get a random sample of 1 and draw it
        # copy samps[1] to crid
        rand_samp = random.choice(samps[2])
        grid = rand_samp.tolist()
        newGrid = np.array(grid, copy=True).flatten()
    elif (keyboard.is_pressed("3")):
        # get a random sample of 1 and draw it
        # copy samps[1] to crid
        rand_samp = random.choice(samps[3])
        grid = rand_samp.tolist()
        newGrid = np.array(grid, copy=True).flatten()
    elif (keyboard.is_pressed("4")):
        # get a random sample of 1 and draw it
        # copy samps[1] to crid
        rand_samp = random.choice(samps[4])
        grid = rand_samp.tolist()
        newGrid = np.array(grid, copy=True).flatten()
    elif (keyboard.is_pressed("5")):
        # get a random sample of 1 and draw it
        # copy samps[1] to crid
        rand_samp = random.choice(samps[5])
        grid = rand_samp.tolist()
        newGrid = np.array(grid, copy=True).flatten()
    elif (keyboard.is_pressed("6")):
        # get a random sample of 1 and draw it
        # copy samps[1] to crid
        rand_samp = random.choice(samps[6])
        grid = rand_samp.tolist()
        newGrid = np.array(grid, copy=True).flatten()
    elif (keyboard.is_pressed("7")):
        # get a random sample of 1 and draw it
        # copy samps[1] to crid
        rand_samp = random.choice(samps[7])
        grid = rand_samp.tolist()
        newGrid = np.array(grid, copy=True).flatten()
    elif (keyboard.is_pressed("8")):
        # get a random sample of 1 and draw it
        # copy samps[1] to crid
        rand_samp = random.choice(samps[8])
        grid = rand_samp.tolist()
        newGrid = np.array(grid, copy=True).flatten()
    elif (keyboard.is_pressed("9")):
        # get a random sample of 1 and draw it
        # copy samps[1] to crid
        rand_samp = random.choice(samps[9])
        grid = rand_samp.tolist()
        newGrid = np.array(grid, copy=True).flatten()
    elif (keyboard.is_pressed("0")):
        # get a random sample of 1 and draw it
        # copy samps[1] to crid
        rand_samp = random.choice(samps[0])
        grid = rand_samp.tolist()
        newGrid = np.array(grid, copy=True).flatten()

    z1 = newGrid.dot(W1)
    a1 = tanh(z1)
    z2 = a1.dot(W2)
    a2 = softmax(z2)
    pygame.display.set_caption(str(np.argmax(a2)))
    if (keyboard.is_pressed("c")):
        grid = []
        for i in range(28):
            grid.append([])
            for j in range(28):
                grid[i].append(0)

    for i in range(28):
        for j in range(28):

            pygame.draw.rect(screen, (grid[i][j] * 255.0, grid[i][j] * 255.0,
                             grid[i][j] * 255.0), [j*CELL_W, i*CELL_W, CELL_W, CELL_W])
    pygame.display.update()
    clock.tick(30)
