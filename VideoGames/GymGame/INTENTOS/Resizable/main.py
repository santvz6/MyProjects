import pygame as pg

from Pantalla import Pantalla

class GoGay:
    def __init__(self):
        pg.init()
        pg.display.set_caption('3T')
        self.clock = pg.time.Clock() 

        self.display = pg.display.set_mode((0,0), pg.FULLSCREEN)
        self.transparente = pg.Surface(pg.display.get_surface().get_size(), pg.SRCALPHA)
        self.pantalla = Pantalla(self.display, self.transparente)



    def update(self):
        while True:
            self.clock.tick(60)
            self.pantalla.update()
            pg.display.update()
            
    

if __name__ == '__main__':
    game = GoGay()
    game.update()