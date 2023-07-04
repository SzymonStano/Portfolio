import math
import pygame
import os
pygame.init()


class Rocket:
    """Obiekt o wspólrzędnych x i y """
    def __init__(self, x: float=0.0,y: float=0.0):
        self.x = x
        self.y = y
       
    def move(self,a,b):        #przesuwanie rakiety o wektor (a,b)
        self.x += float(a)
        self.y += float(b)
        
    def get_position(self):         #wypisuje aktualne położenie rakiety
        return (self.x,self.y)

        
    def get_distance(self,other):       #odległość między dwoma rakietami
        k=self.x - other.x
        n=self.y - other.y
        return math.sqrt(k**2 + n**2)
    
    def __str__(self):
        return "(%s,%s)"% (self.x,self.y)
 
    def __repr__(self):
        return "(%s,%s)"%(self.x,self.y)

width, height = 1000, 600                 #rozmiar wyświetlanego ekranu
window = pygame.display.set_mode((width,height))
pygame.display.set_caption("Rockets!")       
velocity = 0.5   # prędkość jako jednostki przemieszczenia obiektu


def main():

    rakieta1=Rocket(350,150)
    rakieta2=Rocket(700,500)
    
    n=0    #wartośc obrotu w stopniach
    n1=0

    font = pygame.font.SysFont("freesansbold.ttf", 18)   #czcionka do wypisywania położenia

    r1 = pygame.image.load(os.path.join('rakieta.png'))      #obrazek rakiet
    
    R1 = pygame.transform.rotate(pygame.transform.scale(r1, (50,50)),n)  #skalowanie rakiety 1
    R2 = pygame.transform.rotate(pygame.transform.scale(r1, (50,50)),n)  #skalowanie rakiety 2

   
    run = True
    
    while run:

        for event in pygame.event.get():            #uruchamianie i zamykanie programu
            if event.type == pygame.QUIT:

                run = False
                
        położenie1 = font.render("Położenie 1 : " + str(rakieta1), True, (255,0,0))  #renderujemy teksty z położeniami rakiet oraz ich odległośćią od siebie
        położenie2 = font.render("Położenie 2 : " + str(rakieta2), True, (0,255,0))
        odległość = font.render("Odległość 1-2 : " + str(rakieta1.get_distance(rakieta2)), True, (255,255,0))
    
        keys_pressed=pygame.key.get_pressed()   # komendy sterowania i odpowiednia zmiana współrzędnych wraz z komendą
        if keys_pressed[pygame.K_LEFT] and rakieta1.x>0 :         #sterujemy strzałkami
            rakieta1.x -= velocity
            R1 = pygame.transform.rotate(pygame.transform.scale(r1, (50,50)),90)  #zmiana ustawienia rakiety względem ruchu
        if keys_pressed[pygame.K_RIGHT] and rakieta1.x<(width-50):        #ustawione ograniczenia położenia za warunkiem and
            rakieta1.x += velocity
            R1 = pygame.transform.rotate(pygame.transform.scale(r1, (50,50)),270)
        if keys_pressed[pygame.K_UP] and rakieta1.y>0:
            rakieta1.y -= velocity
            R1 = pygame.transform.rotate(pygame.transform.scale(r1, (50,50)),0)
        if keys_pressed[pygame.K_DOWN] and rakieta1.y<(height-50):
            rakieta1.y += velocity
            R1 = pygame.transform.rotate(pygame.transform.scale(r1, (50,50)),180)

        if keys_pressed[pygame.K_a] and rakieta2.x>0:         #drugą rakietą sterujemy "wsad"
            rakieta2.x -= velocity
            R2 = pygame.transform.rotate(pygame.transform.scale(r1, (50,50)),90)
        if keys_pressed[pygame.K_d] and rakieta2.x<(width-50):
            rakieta2.x += velocity
            R2 = pygame.transform.rotate(pygame.transform.scale(r1, (50,50)),270)
        if keys_pressed[pygame.K_w] and rakieta2.y>0:
            rakieta2.y -= velocity
            R2 = pygame.transform.rotate(pygame.transform.scale(r1, (50,50)),0)
        if keys_pressed[pygame.K_s] and rakieta2.y<(height-50):
            rakieta2.y += velocity
            R2 = pygame.transform.rotate(pygame.transform.scale(r1, (50,50)),180)
    
        window.fill((100,10,255))     #kolor ekranu
        window.blit(położenie1, (50,50) )    #wyświetlone położenia i odległóść
        window.blit(położenie2, (50,60) )
        window.blit(odległość, (50,70) )
        window.blit(R1, (rakieta1.x,rakieta1.y))      #wyświetlona rakieta
        window.blit(R2, (rakieta2.x,rakieta2.y))
        pygame.display.update()     #aktualizowanie ekranu

        # while rakieta1.get_distance(rakieta2) < 50:
        #     window.blit(pygame.image.load(os.path.join('explosion.png'), ((rakieta1.x + rakieta2.x)/2, (rakieta1.y + rakieta2.y)/2))


    pygame.quit()


if __name__=="__main__":
    main()
