import numpy as np
from .binary_perceptron import BinaryPerceptron


if __name__ == "__main__":
    # Exec:     python3 -m chapters.c2.main
    

    def test(X, y):
        p = BinaryPerceptron(learning_rate= 0.1)
        errors = p.fit(X, y).errors_
        print(errors)
        
    # AND dataset (LINEAL)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([-1, -1, -1, 1])
    test(X, y)

    # OR dataset (LINEAL)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
        ])
    y = np.array([-1, 1, 1, 1])
    test(X, y)

    # XOR dataset (NO LINEAL)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([-1, 1, 1, -1])
    test(X, y)