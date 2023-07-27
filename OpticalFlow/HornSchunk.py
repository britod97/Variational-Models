# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal

def convolve(u, filter_size=50):
    t = np.linspace(-10, 10, filter_size)
    bump = np.exp(-0.1*t**2)
    bump /= np.trapz(bump) # normalize the integral to 1
    # make a 2-D kernel out of it
    kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
    # mode='same' is there to enforce the same output shape as input arrays
    img3 = signal.fftconvolve(u, kernel, mode='same')
    return(img3)


def partialDerivs(E, E1):
    # Calculates the partial derivatives of the given list of frames
    M, N = frames[0].shape
    X = np.zeros((M,N))
    Y = np.zeros((M,N))
    T = np.zeros((M,N))
    for i in range(M-1):
        for j in range(N-1):
            Ex = ( E[i,j+1]    - E[i,j]
                  + E[i+1,j+1]  - E[i+1,j]
                  + E1[i,j+1]   - E1[i,j]
                  + E1[i+1,j+1] - E1[i+1,j] )*(1/4)
            Ey = ( E[i+1,j]    - E[i,j]
                  + E[i+1,j+1]  - E[i,j+1]
                  + E1[i+1,j]   - E1[i,j]
                  + E1[i+1,j+1] - E1[i,j+1] )*(1/4)
            Et = ( E1[i,j]     - E[i,j]
                  + E1[i+1,j]   - E[i+1,j]
                  + E1[i,j+1]   - E[i,j+1]
                  + E1[i+1,j+1] - E[i+1,j+1] )*(1/4)
            X[i,j] = Ex
            Y[i,j] = Ey
            T[i,j] = Et
    return(X, Y, T)


def calculateBars(matrix):
    # Returns the weighted averages described in the paper given a matrix
    M, N = matrix.shape
    Ubar = np.zeros((M,N))
    for x in range(1,M-1):
        for y in range(1,N-1):
            Ubar[x,y] = ( 1/12*(matrix[x-1,y-1] + matrix[x+1,y-1] + matrix[x-1,y+1] + matrix[x+1,y+1])
                        + 1/6*(matrix[x-1,y] + matrix[x,y-1] + matrix[x+1,y] + matrix[x,y+1]) - matrix[x,y] )
    return(Ubar)


def nextIteration(U, V, Ubar, Vbar, Ex, Ey, Et, alpha):
    # Returns the next iteration when given the matrices for U, V, and partial derivatives
    # This is to be run on simply one frame
    M,N = U.shape
    new_U, new_V = np.zeros((M,N)), np.zeros((M,N))
    for x in range(1,M-1):
        for y in range(1,N-1):
            denom = 4*(alpha**2) + (Ex[x,y])**2 + (Ey[x,y])**2
            numerator_U = Ex[x,y]*(Ex[x,y]*Ubar[x,y] + Ey[x,y]*Vbar[x,y] + Et[x,y])
            numerator_V = Ey[x,y]*(Ex[x,y]*Ubar[x,y] + Ey[x,y]*Vbar[x,y] + Et[x,y])
            new_U[x,y] = Ubar[x,y] - numerator_U/denom
            new_V[x,y] = Vbar[x,y] - numerator_V/denom
    return(new_U, new_V)

def estimateFlowVectors(frame0, frame1, N_iter, alpha):
    M,N = frame0.shape
    Ex, Ey, Et = partialDerivs(frame0, frame1)
    U, V = np.zeros((M,N)), np.zeros((M,N))
    for k in range(N_iter):
        Ubar, Vbar = calculateBars(U), calculateBars(V)
        U, V = nextIteration(U, V, Ubar, Vbar, Ex, Ey, Et, alpha)
    return(U, V)

def readFrames(folderpath):
    filenames = os.listdir(folderpath)
    frames = []
    for FILE in filenames:
        filepath = folderpath + '/' + FILE
        img = np.array( Image.open(filepath).convert('L') , dtype='int64')
        frames.append(img)
    return(frames)


#compute magnitude in each 8 pixels. return magnitude average
def get_magnitude(u, v):
    scale = 3
    sum = 0.0
    counter = 0.0

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            counter += 1
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            sum += magnitude

    mag_avg = sum / counter

    return mag_avg


def draw_quiver(u,v,beforeImg):
    scale = 1
    ax = plt.figure().gca()
    ax.imshow(beforeImg, cmap = 'gray')

    magnitudeAvg = get_magnitude(u, v)

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            #draw only significant changes
            if magnitude > magnitudeAvg:
                ax.arrow(j,i, dx, dy, color = 'red')
    
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    plt.draw()
    plt.show()


if __name__ == '__main__':
    frames = readFrames('eval-data/Dumptruck')
    
    L = len(frames)
    for i in range(L-1):
        print(i)
        frame0 = convolve(frames[i], filter_size=5)/255
        frame1 = convolve(frames[i+1], filter_size=5)/255
        
        N_iter = 32
        alpha = 0.5
        
        Ex, Ey, Et = partialDerivs(frame0, frame1)
        U1, V1 = estimateFlowVectors(frame0, frame1, N_iter, alpha)
        
        M,N = frame0.shape
         
        draw_quiver(U1, V1, frame0)
    