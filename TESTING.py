import sys
import pygame
import win32api
import pickle
import numpy as np
import keyboard
import random
import pandas as pd


data_name = input("What is the name of your dataset (Excluding .csv): ")


# This is more interesting
pygame.init()
df = pd.read_csv("datasets/"+data_name+".csv")["label"].unique().tolist()
if data_name== "MNIST":df = sorted(df)
print(df)
with open("weights/W1_" + str(data_name)+".bin", "rb") as f:
    W1 = pickle.load(f)
with open("weights/W2_" + str(data_name)+".bin", "rb") as g:
    W2 = pickle.load(g)


def tanh(x, deriv=False):
    if deriv:
        return 1.0 - (np.power(x, 2))
    return np.tanh(x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


CELL_W = 15
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
        if event.type == pygame.QUIT:
            sys.exit()
    if win32api.GetKeyState(0x01) < 0:
        mX, mY = pygame.mouse.get_pos()
        i = mY//CELL_W
        j = mX//CELL_W
        grid[i][j] += random.uniform(0.2, 0.6)
        grid[i][j] = min(1.0, grid[i][j])
        if i >= 1:
            grid[i-1][j] += random.uniform(0.05, 0.2)
            grid[i-1][j] = min(1.0, grid[i-1][j])
        if (i < 27):
            grid[i+1][j] += random.uniform(0.05, 0.2)
            grid[i+1][j] = min(1.0, grid[i+1][j])
        if j >= 1:
            grid[i][j-1] += random.uniform(0.05, 0.2)
            grid[i][j-1] = min(1.0, grid[i][j-1])
        if (j < 27):
            grid[i][j+1] += random.uniform(0.05, 0.2)
            grid[i][j+1] = min(1.0, grid[i][j+1])

        newGrid = np.array(grid, copy=True).flatten()
    z1 = newGrid.dot(W1)
    a1 = tanh(z1)
    z2 = a1.dot(W2)
    a2 = softmax(z2)
    pygame.display.set_caption("I predict:  " + str(df[np.argmax(a2)]))
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
    clock.tick(60)
