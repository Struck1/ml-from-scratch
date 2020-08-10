import numpy as np
from nnfs.datasets import spiral_data


X, y = spiral_data(100, 3)

print(X.shape)
print(y.shape)

class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        #init parametres
        n_samples, n_features = X.shape
        self.weights =np.zeros(n_features)
        
        self.bias = 0
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
           
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))   
            db = (1/n_samples) * np.sum(y_predicted -y)

            self.weights -= self.lr *dw 
            self.bias -= self.lr *db
            
     

    def predic(self, X):
         print(self.weights, self.bias)
         y_predicted = np.dot(X, self.weights) + self.bias 
         print(f'x test : {X} predict : {y_predicted}')
         return y_predicted





# X = np.array([[1, 2, 3, 2.5],
#      [2.0, 5.0, -1.0, 2.0],
#      [-1.5, 2.7, 3.3, -0.8]])


class Layer():
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10* np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))
    
    def foward(self, inputs):

        self.output = np.dot(inputs, self.weights) + self.bias


class Active:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)





