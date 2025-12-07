import pygame as pg

class Jugador:
    def __init__(self, nombre:str, estadisticas:dict) -> None:
        ANCHO, ALTO = pg.display.get_surface().get_size()
        
        #self.proporcion = 1 if self.proporcion > 1000 else 1/3

        self.rect = pg.Rect(ANCHO // 2, ALTO // 2, 50, 50)
        self.velocidad = 5 # trabajamos con enteros, sino falla



"""
Al hacer pequeña la pantalla creamos una nueva instancia?

"""