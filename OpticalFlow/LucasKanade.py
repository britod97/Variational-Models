# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal

def convolve(u, filter_size=20):
    t = np.linspace(-10, 10, filter_size)
    bump = np.exp(-0.1*t**2)
    bump /= np.trapz(bump) # normalize the integral to 1
    # make a 2-D kernel out of it
    kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
    # mode='same' is there to enforce the same output shape as input arrays
    img3 = signal.fftconvolve(u, kernel, mode='same')
    return(img3)


def makeWindowAtPoint(p, img, window_size):
    x = p[0]
    y = p[1]
    window = np.zeros((2*window_size+1, 2*window_size+1))
    for m in range(-window_size, window_size+1):
        for n in range(-window_size, window_size+1):
            i=m+window_size
            j=n+window_size
            window[i,j] = img[x+m][y+n]
    return(window)

def calcFlowAtPoint(p, img0, img1, window_size):
    window0 = makeWindowAtPoint(p, img0, window_size)
    window1 = makeWindowAtPoint(p, img1, window_size)
    grad = np.gradient(window0)
    Ix = np.array(grad[0]).flatten()
    Iy = np.array(grad[1]).flatten()
    It = np.array(window0 - window1).flatten()
    
    A = np.array([Ix.T, Iy.T])
    b = -It
    
    v1 = np.dot(A, A.T)
    v1inv = np.linalg.inv(v1)
    v2 = np.dot(v1inv, A)
    v = np.dot(v2, b.T)
    return(v)
    
def LucasKanade(img0, img1, window_size):
    M, N = img0.shape
    U, V = np.zeros((M,N)), np.zeros((M,N))
    for x in range( window_size, M-window_size ):
        for y in range( window_size, N-window_size ):
            p = (x,y)
            v = calcFlowAtPoint(p, img0, img1, window_size)
            U[x,y] = v[0]
            V[x,y] = v[1]
    return(U, V)
    

if __name__ == '__main__':
    frame0 = convolve(Image.open('C:/Users/brito/Desktop/MCC/Variational Models/OpticalFlow/eval-data/Backyard/frame07.png').convert('L'))
    frame1 = convolve(Image.open('C:/Users/brito/Desktop/MCC/Variational Models/OpticalFlow/eval-data/Backyard/frame08.png').convert('L'))
    
    U, V = LucasKanade(frame0, frame1, 3)
    
    fig1, ax1 = plt.subplots(1,1)
    ax1.imshow(U, cmap='gray')
    
    fig2, ax2 = plt.subplots(1,1)
    ax2.imshow(V, cmap='gray')
    
    plt.show()
    
    
    
    