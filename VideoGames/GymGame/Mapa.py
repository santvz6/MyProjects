import pygame as pg

class Mapa:
    def __init__(self, Pantalla) -> None:
        self.Pantalla = Pantalla
        self.velocidad = self.Pantalla.jugador.velocidad 
        self.ANCHO, self.ALTO = pg.display.get_surface().get_size()

        self.imagen = pg.transform.scale(pg.image.load('TILED/MAP_TMX.png'), (1920*5/2, 1280*5/2))
        self.posicion = [0, 0]
        self.angulo = 0


    def movimiento(self,keys):
        if keys[pg.K_LEFT]:
            self.posicion[0] += self.velocidad
            self.angulo = 180
        if keys[pg.K_RIGHT]:
            self.posicion[0] -= self.velocidad
            self.angulo = 0
        if keys[pg.K_UP]:
            self.posicion[1] += self.velocidad
            self.angulo = 90
        if keys[pg.K_DOWN]:
            self.posicion[1] -= self.velocidad
            self.angulo = 270
        
        return self.angulo

    def colision(self, keys):
        if keys[pg.K_LEFT]:
            self.posicion[0] -= self.velocidad
            self.angulo = 180
        if keys[pg.K_RIGHT]:
            self.posicion[0] += self.velocidad
            self.angulo = 0
        if keys[pg.K_UP]:
            self.posicion[1] -= self.velocidad
            self.angulo = 90
        if keys[pg.K_DOWN]:
            self.posicion[1] += self.velocidad
            self.angulo = 270
        
        return self.angulo
        
    
    def paredes(self) -> list:
        grosor = 20
        return [pg.Rect(0, 0, self.ANCHO, grosor),                  # Pared superior
            pg.Rect(0, 0, grosor, self.ALTO),                       # Pared izquierda
            pg.Rect(0, self.ALTO - grosor, self.ANCHO, grosor),     # Pared inferior
            pg.Rect(self.ANCHO - grosor, 0, grosor, self.ALTO)]     # Pared derecha