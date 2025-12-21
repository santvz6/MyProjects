import os

import numpy as np
import matplotlib.pyplot as plt


class BinaryPerceptron:
    """
    Remember that only can classify linearly separable data
    """

    def __init__(self, learning_rate: float, num_iter= 1000, seed= 1):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.seed = seed
    

    def fit(self, X, y):
        rgen = np.random.RandomState(self.seed)
        self.w_ = rgen.normal(loc= 0, scale= 0.1, size= X.shape[1] + 1) # size = num_features + 1(bias)
        self.errors_ = []

        for i in range(self.num_iter): 
            errors = 0
            print(self.w_)

            for x, target in zip(X, y):  
                update = self.learning_rate * (target - self.predict(x))

                self.w_[0] += update
                self.w_[1:] += update * x

                # Guardamos errores
                errors += int(update != 0)

            self.errors_.append(errors)

        return self
    
    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]
    

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)
    

if __name__ == "__main__":
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([-1, -1, -1, 1])


    p = BinaryPerceptron(learning_rate= 0.1, seed= 2)
    errors = p.fit(X, y).errors_

