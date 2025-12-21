import numpy as np
from scipy.signal import convolve2d

from logger_config import logger



class ConwayGame:
    def __init__(self, state, seed, size=32):
        np.random.seed(seed)
        self.size = size
        self.state = state

    def update(self):
        """Calcula el siguiente estado basado en las reglas de Conway."""
        # Kernel para contar vecinos (excluyendo la celda central)
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])
        
        # Contamos cuántos vecinos vivos tiene cada celda
        neighbors = convolve2d(self.state, kernel, mode="same", boundary="wrap")
        
        # Aplicamos las reglas
        next_state = np.zeros((self.size, self.size), dtype=int)
        
        # Sobreviven o nacen
        next_state[(self.state == 1) & ((neighbors == 2) | (neighbors == 3))] = 1
        next_state[(self.state == 0) & (neighbors == 3)] = 1
        
        self.state = next_state
        return self.state

    def generate_sequence(self, steps):
        """Genera una serie de estados (útil para series temporales)."""
        history = [self.state.copy()] # shape(steps+1, size, size)
        for _ in range(steps):
            history.append(self.update().copy())
        return np.array(history)
    
