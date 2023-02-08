import pygame
import win32api
import pickle
import numpy as np
import matplotlib.pyplot as plt
import keyboard

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
        x[x<0]=0
        x[x>0]=1
        return x
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)




CELL_W = 20
screen = pygame.display.set_mode((28 * CELL_W, 28 * CELL_W))
grid = []
for i in range(28):
    grid.append([])
    for j in range(28):
        grid[i].append(0)
while True:
    screen.fill((0,0,0))
    for event in pygame.event.get():
        if event.type == pygame.MOUSEMOTION and win32api.GetKeyState(0x01)<0:
            mX, mY = pygame.mouse.get_pos()
            i = mY//CELL_W
            j = mX//CELL_W
            grid[i][j] += 0.2
            grid[i][j] = min(1.0, grid[i][j])


    newGrid = np.array(grid,copy=True).flatten()
    z1 = newGrid.dot(W1)
    a1 = tanh(z1)
    z2 = a1.dot(W2)
    a2 = softmax(z2)
    pygame.display.set_caption(str(np.argmax(a2)))    
    if(keyboard.is_pressed("c")):
        plt.imshow(grid)
        plt.show()
    
    for i in range(28):
        for j in range(28):
            pygame.draw.rect(screen, (grid[i][j] * 255.0,grid[i][j] * 255.0,grid[i][j] * 255.0), [j*CELL_W,i*CELL_W,CELL_W,CELL_W])
    pygame.display.update()