import pygame as pg

from Pantalla import Pantalla

class GoGay:
    def __init__(self):
        pg.init()
        pg.display.set_caption('3T')
        self.clock = pg.time.Clock() 

        self.ANCHO = 1280
        self.ALTO = 720
        self.display = pg.display.set_mode((self.ANCHO,self.ALTO))
        
        self.transparente = pg.Surface(pg.display.get_surface().get_size(), pg.SRCALPHA)
        self.dt = self.clock.tick(60)

        self.pantalla = Pantalla(self)



    def update(self):
        while True:
            self.dt = self.clock.tick(60)
            self.pantalla.update()
            pg.display.update()
            
    

if __name__ == '__main__':
    game = GoGay()
    game.update()