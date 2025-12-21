import pygame as pg
import sys

from Jugador import Jugador
from Colisiones import Colisiones


class Pantalla:
    def __init__(self, display, transparente):
        self.display = display
        self.transparente = transparente

        self.jugador = Jugador('Rodrigo', {})
        self.colisiones = Colisiones(self.display)

        self.mapa_pos = 0
        self.proporcion = 1

    def update(self):
        for event in pg.event.get():
            if event.type == pg.QUIT: 
                sys.exit()

            #Evento de tipo Click Izquierdo
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:  
                m_pos = pg.mouse.get_pos()

            if event.type == pg.KEYDOWN:
                if event.key == pg.K_0:
                    self.display = pg.display.set_mode((1920*2/3, 1080*2/3))
                    self.proporcion = 2/3

            
        keys = pg.key.get_pressed()
        if keys[pg.K_LEFT]:
            self.mapa_pos += self.jugador.velocidad
        if keys[pg.K_RIGHT]:
            self.mapa_pos -= self.jugador.velocidad
        if keys[pg.K_UP]:
            self.jugador.rect.y -= self.jugador.velocidad
        if keys[pg.K_DOWN]:
            self.jugador.rect.y += self.jugador.velocidad

        self.display.fill((0,0,0))

        paredes = self.colisiones.paredes()

        for pared in paredes:
            pared_ajustada = pg.draw.rect(self.display, (0,255,0), pg.Rect((pared.x + self.mapa_pos), pared.y, pared.width, pared.height))
        
        if self.jugador.rect.colliderect(pared_ajustada):
            if keys[pg.K_LEFT]:
                self.mapa_pos += self.jugador.velocidad
            if keys[pg.K_RIGHT]:
                self.mapa_pos -= self.jugador.velocidad
            if keys[pg.K_UP]:
                self.jugador.rect.y -= self.jugador.velocidad
            if keys[pg.K_DOWN]:
                self.jugador.rect.y += self.jugador.velocidad
        
        pg.draw.rect(self.display, (255,0,0), tuple(_ * self.proporcion for _ in self.jugador.rect))