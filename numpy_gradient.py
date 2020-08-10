import numpy as np


X = np.array([1,2,3,4], dtype=np.float32)

Y = np.array([2,4,6,8], dtype=np.float32)

w = 0

#model prediction

def foward(x):
    return  np.dot(x,w)

#loss
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

#gradient

def gradient(x, y_pred, y):

    return np.dot(2*x, y_pred - y).mean()


print(f'Prediction before traning: f(5) = {foward(5):.3f}')

#Traning

learning_rate = 0.01
n_iter = 10


for epoch in range(n_iter):
    #prediction forward 

    y_pred = foward(X)

    #loss 

    loss1 = loss(Y, y_pred)

    #gradient

    dw = gradient(X, y_pred, Y)

    #update weigths

    w -= learning_rate * dw
    
    if epoch % 1 == 0:
        print(f'epoch {epoch +1 } : w = {w:.3f}, loss = {loss1: .8f}')

print(f"Prediction after traning: f(5) = {foward(5):.3f}")    
