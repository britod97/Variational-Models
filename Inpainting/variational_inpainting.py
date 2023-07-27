import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rndm
from PIL import Image
from scipy import signal

def divergence(u, delta=0.00001):    # delta is a small term used to avoid division by 0
    grad_u = np.gradient(u)
    grad_u = grad_u/ (np.sqrt(grad_u[0]**2 + grad_u[1]**2 + delta) ) # divide by the norm of the gradient
    
    ux = grad_u[0]
    uy = grad_u[1]                  # ux and uy are the first partial derivatives
    uxx = np.gradient(ux)[0]
    uyy = np.gradient(uy)[1]        # uxx and uyy are the second partial derivatives
    div = uxx + uyy
    return(div)

def makeLambdaArray(corrupted, lam):
    NA = np.where(corrupted == 255)
    X = NA[0]
    Y = NA[1]
    arr = np.full(corrupted.shape, lam)
    for i in range(len(X)):
        x, y = X[i], Y[i]
        arr[x,y] = 0
    return(arr)

def grad_descent(u0, corrupted, N_iter, lam, alpha):
    u_old = u0
    LamArr = makeLambdaArray(corrupted, lam)
    for i in range(N_iter):
        div = divergence(u_old)
        gradF = -div + LamArr*(u0 - u_old)
        
        if i % 100 == 0:
            print('Norma =', np.linalg.norm(gradF))
        
        u_new = u_old - alpha*gradF
        u_old = u_new
    return(u_new)

if __name__ == '__main__':
    corrupted = np.array(Image.open("C:/Users/brito/Desktop/MCC/Variational Models/imgs/cameraman_corrupted2.png").convert('L'))
    M, N = corrupted.shape
    u0 = np.copy(corrupted)
    for i in range(M):
        for j in range(N):
            if u0[i,j] == 255:
                u0[i,j] = np.random.randint(0,255)
    
    u0 = u0/255
    
    N_iter = 500
    lam = 0.001
    alpha = 0.001
    print('N_iter = ', N_iter, ' lam = ', lam, ' alpha = ', alpha)
    
    img_new = grad_descent(u0, corrupted, N_iter, lam, alpha)
    
    plt.imshow(img_new, cmap='gray')