import pygame as pg
import sys
import time

from Jugador import Jugador
from Mapa import Mapa


class Pantalla():
    def __init__(self,GoGay):

        self.GoGay = GoGay
        self.display = self.GoGay.display
        self.transparente = self.GoGay.transparente
        self.dt = self.GoGay.dt

        self.jugador = Jugador(self, 'Rodrigo', {}, 1)
        self.mapa = Mapa(self)

        
        self.frame_time = 0
        self.tiempo_transcurrido = 0
        


    def update(self):
        self.display.fill((0, 0, 0))  # Limpia la pantalla llenándola de negro

        for event in pg.event.get():
            if event.type == pg.QUIT: 
                sys.exit()

            #Evento de tipo Click Izquierdo
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:  
                m_pos = pg.mouse.get_pos()
        
        self.tiempo_transcurrido += self.dt

        if self.tiempo_transcurrido > 160:
            self.frame_time = (self.frame_time + 1) % 8
            self.tiempo_transcurrido = 0


        #self.frame_time = time.t
        keys = pg.key.get_pressed()
        self.jugador.rotar(self.mapa.movimiento(keys))

        self.display.blit(self.mapa.imagen,(self.mapa.posicion[0], self.mapa.posicion[1]))


        #paredes = self.mapa.paredes()
        paredes = []

        for pared in paredes:
            pared_ajustada = pg.draw.rect(self.display, (0,255,0), pg.Rect((pared.x + self.mapa.posicion[0]),(pared.y + self.mapa.posicion[1]), pared.width, pared.height))
        
            if self.jugador.rect.colliderect(pared_ajustada):
                self.mapa.colision(keys)

        if keys[pg.K_LEFT] or keys[pg.K_RIGHT] or keys[pg.K_UP] or keys[pg.K_DOWN]:
            self.jugador.enMovimiento()
        else:
            self.jugador.idle()
            
            