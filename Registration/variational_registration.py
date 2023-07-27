# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import laplace

def A(U1, U2):
    return( (laplace(U1), laplace(U2)) )

def f(x, y, U1, U2, R, T):
    M,N = R.shape
    
    Tx, Ty = np.gradient(T)
    u1, u2 = U1[x,y], U2[x,y]
    
    if (x-round(u1) < 0) or (x-round(u1) >= M) or (y-round(u2) < 0) or (y-round(u2) >= M):
        # If "coming" from outside the image domain, we simply write a 0
        T_pixel = 0
    
    else:
        T_pixel = T[x-round(u1), y-round(u2)]
    
    output_part1 = R[x,y] - T_pixel    # This is a number
    output_part2 = np.array([ Tx[x,y], Ty[x,y] ])   # This is an array of two numbers
    output = output_part1*output_part2
    return(output)

def grad_descent(N_iter, lamb, alpha, R, T):
    M, N = R.shape
    U1, U2 = np.zeros((M,N)), np.zeros((M,N))
    for i in range(N_iter):
        print('Iteration', i)
        A1, A2 = A(U1, U2)
        for x in range(M):
            for y in range(N):
                f1, f2 = f(x, y, U1, U2, R, T)
                U1[x,y] = U1[x,y] - alpha * (f1 + lamb*A1[x,y])
                U2[x,y] = U2[x,y] - alpha * (f2 + lamb*A2[x,y])
    return(U1, U2)

def moveReferenceImage(R, U1, U2):
    M, N = R.shape
    new_image = np.zeros((M,N))
    for x in range(M):
        for y in range(N):
            u1, u2 = U1[x,y], U2[x,y]
            if (x-round(u1) < 0) or (x-round(u1) >= M) or (y-round(u2) < 0) or (y-round(u2) >= M):
                # If "coming" from outside the image domain, we simply write a 0
                R_pixel = 0
            
            else:
                R_pixel = T[x-round(u1), y-round(u2)]
            new_image[x,y] = R_pixel
    return(new_image)
            

if __name__ == '__main__':
    temp = np.array( Image.open("template.png").convert('L') )
    ref = np.array( Image.open("reference.png").convert('L') )
    
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(temp,cmap='gray')
    axs[0].set_title('Template')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    
    axs[1].imshow(ref,cmap='gray')
    axs[1].set_title('Reference')
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    
    N_iter = 10
    alpha = 0.001
    lamb = 0.001
    R = ref
    T = temp
    U1, U2 = grad_descent(N_iter, lamb, alpha, R, T)