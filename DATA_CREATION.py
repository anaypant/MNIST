import sys
import pygame
import win32api
import numpy as np
import matplotlib.pyplot as plt
import keyboard
from keras.datasets import mnist
import random
import pandas as pd
import time
from ast import literal_eval


# -- boring shit, don't look at this

# This is more interesting

dataset_name = input("Name your data: ")


pygame.init()
df = pd.DataFrame({"data":[],"label":[]})
CELL_W = 15
pen_size=2
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
        if pen_size==2:
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

    if (keyboard.is_pressed("1")):
        pen_size=1
    elif keyboard.is_pressed("2"):
        pen_size=2

    if (keyboard.is_pressed("c")):
        grid = []
        for i in range(28):
            grid.append([])
            for j in range(28):
                grid[i].append(0)

    if keyboard.is_pressed("enter"):
        # add data to dataframe
        bucket = input("Label of image: ")
        while(bucket == ""):
            bucket = input("Label of image: ")
        df.loc[len(df.index)] = [newGrid.tolist(), bucket]
        print("Added")
        grid = []
        for i in range(28):
            grid.append([])
            for j in range(28):
                grid[i].append(0)
        time.sleep(0.5)
        

    if keyboard.is_pressed("esc"):
        df.to_csv("datasets/"+str(dataset_name)+".csv",index=False,header=["data","label"])
        quit()
    for i in range(28):
        for j in range(28):

            pygame.draw.rect(screen, (grid[i][j] * 255.0, grid[i][j] * 255.0,
                             grid[i][j] * 255.0), [j*CELL_W, i*CELL_W, CELL_W, CELL_W])
    pygame.display.update()
    clock.tick(60)
