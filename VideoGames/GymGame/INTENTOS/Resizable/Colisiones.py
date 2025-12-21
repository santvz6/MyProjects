import pygame as pg

class Colisiones:
    def __init__(self, display) -> None:
        self.display = display

        
    def paredes(self) -> list:
        self.ANCHO, self.ALTO = pg.display.get_surface().get_size()
        self.proporcion = 1 if self.display.get_flags() & pg.FULLSCREEN else 2/3

        grosor = 20 * self.proporcion
        
        print(self.ANCHO, self.ALTO)
        return [pg.Rect(0, 0, self.ANCHO, grosor),  # Pared superior
            pg.Rect(0, 0, grosor, self.ALTO),  # Pared izquierda
            pg.Rect(0, self.ALTO - grosor, self.ANCHO, grosor),  # Pared inferior
            pg.Rect(self.ANCHO - grosor, 0, grosor, self.ALTO)]  # Pared derecha