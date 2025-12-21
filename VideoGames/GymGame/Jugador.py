import pygame as pg

class Jugador():
    def __init__(self, Pantalla, nombre:str, estadisticas:dict, numImagen:int) -> None:
        
        self.Pantalla = Pantalla
        self.display = self.Pantalla.GoGay.display
        self.ANCHO, self.ALTO = pg.display.get_surface().get_size()
        

        self.nombre = nombre
        self.estadisticas = estadisticas
        self.numImagen = numImagen
        self.imagen = self.getImagen()
        
        
        self.rect = pg.Rect(self.ANCHO // 2, self.ALTO // 2, 128, 128)
        self.velocidad: int = 4 #estadisticas['velocidad']
    
    def getImagen(self) -> pg.image:
        match self.numImagen:
            case 1:
                return pg.transform.scale(pg.image.load('Imagenes/Jugador/sprite_sheet.png'),(1024,128))

    def idle(self):
        self.display.blit(pg.transform.rotate(self.getImagen(), self.angulo),(self.ANCHO//2,self.ALTO//2),
                            pg.Rect(0, 0, 128, 128))

    def enMovimiento(self):
        self.display.blit(self.imagen, (self.ANCHO//2,self.ALTO//2))

    def rotar(self, angulo):
        self.imagen = pg.transform.rotate(self.getImagen().subsurface(pg.Rect(self.Pantalla.frame_time*128, 0, 128, 128)), angulo)
        self.angulo = angulo

    
        