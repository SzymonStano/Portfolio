import numpy as np

class Empty(Exception):
    pass
class Queue:
    DEFAULT_CAPACITY = 10

    def __init__(self):
        self._data = [None] * Queue.DEFAULT_CAPACITY
        self._size = 0
        self._front = 0

    def __len__(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def first(self):
        if self.is_empty():
            raise Empty('Queue is empty')
        return self._data[self._front]

    def dequeue(self):
        if self.is_empty():
            raise Empty('Queue is empty')
        value = self._data[self._front]
        self._data[self._front] = None
        self._front = (self._front + 1) % len(self._data)
        self._size -= 1
        return value

    def enqueue(self, e):
        if self._size == len(self._data):
            self._resize(2 * len(self._data))
        avail = (self._front + self._size) % len(self._data)
        self._data[avail] = e
        self._size += 1

    def _resize(self, cap):
        old = self._data
        self._data = [None] * cap
        walk = self._front
        for k in range(self._size):  # only existing elements
            self._data[k] = old[walk]
            walk = (1 + walk) % len(old)
        self._front = 0


class Vertex:
    # """Klasa Węzłów zawierającaw w sobie: Id jako numer węzła, """
    def __init__(self, num):
        self.id = num
        self.connectedTo = {}       # list of neighbours, and their weights of connections! (stored as map)
        self.color = 'white'        # color of node
        # self.dist = sys.maxsize     # distance from beginning (will be used later) WHYYY
        self.dist = 0
        self.pred = None            # predecessor
        self.disc = 0               # discovery time
        self.fin = 0                # end-of-processing time
        self.move = 0               # NOWOŚĆ!!!!

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def setColor(self, color):
        self.color = color

    def setDistance(self, d):
        self.dist = d

    def setPred(self, p):
        self.pred = p

    def setDiscovery(self, dtime):
        self.disc = dtime

    def setFinish(self, ftime):
        self.fin = ftime

    def setMove(self,move):
        self.move = move

    def getFinish(self):
        return self.fin

    def getDiscovery(self):
        return self.disc

    def getPred(self):
        return self.pred

    def getDistance(self):
        return self.dist

    def getColor(self):
        return self.color

    def getConnections(self):
        return self.connectedTo.keys()

    def getWeight(self, nbr):
        return self.connectedTo[nbr]

    def getMove(self):
        return self.move

    # def __str__(self):              # lekko zmienione
    #     return str(self.id) + ":color " + self.color + ":disc " + str(self.disc) + ":fin " + str(
    #         self.fin) + ":dist " + str(self.dist) + ":pred \n\t[" + str(self.pred) + "]\n"
    def __str__(self):
        return str(self.id) + "  color:" + self.color + "  disc:" + str(self.disc) + "  fin:" + str(
            self.fin) + "  dist:" + str(self.dist) + "  pred:\t[" + str(self.pred) + "]"



    def getId(self):
        return self.id


class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0
        self.time = 0

    def addVertex(self,key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)        # Wprowadzony klucz węzła do listy w grafie, jest również jego Id w klasie Vertx
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self,n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self,n):
        return n in self.vertList

    def addEdge(self,f,t,cost=0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], cost)

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())

    # def __copy__(self):
    #     return copy.copy(self)


    # Zadanie 3
    # Breadth First Search
    def bfs(self, start):
        start.setDistance(0)  # distance 0 indicates it is a start node
        start.setPred(None)  # no predecessor at start
        vertQueue = Queue()
        vertQueue.enqueue(start)  # add start to processing queue
        while (vertQueue._size > 0):
            currentVert = vertQueue.dequeue()  # pop next node to process -> current node
            for nbr in currentVert.getConnections():  # check all neighbors of the current node
                if (nbr.getColor() == 'white'):  # if the neighbor is white
                    nbr.setColor('gray')  # change its color to grey
                    nbr.setDistance(currentVert.getDistance() + 1)  # set its distance
                    nbr.setPred(currentVert)  # current node is its predecessor
                    vertQueue.enqueue(nbr)  # add it to the queue
            currentVert.setColor('black')  # change current node to black after visiting all of its neigh.

    # Depth First Search
    def dfs(self):
        for aVertex in self:
            aVertex.setColor('white')
            aVertex.setPred(None)
        for aVertex in self:
            if aVertex.getColor() == 'white':
                self.dfsvisit(aVertex)

    def dfsvisit(self, startVertex):
        startVertex.setColor('gray')
        self.time += 1
        startVertex.setDiscovery(self.time)
        for nextVertex in startVertex.getConnections():
            if nextVertex.getColor() == 'white':
                nextVertex.setPred(startVertex)
                self.dfsvisit(nextVertex)
        startVertex.setColor('black')
        self.time += 1
        startVertex.setFinish(self.time)

    def traverse(self, y):
        x = y
        way=[]
        while (x.getPred()):
            way.append(x.getId())
            x = x.getPred()
        way.append(x.getId())
        return way
    #
    # def traverse(self, y):
    #     for dist


L = np.array([3, 3, 1])   # represented as number of Missionaries, Cannibals, and presence of boat on certain riverbank
P = ([3, 3, 1]) - L
left_limit = np.array([0,0,0])
right_limit = np.array([3,3,1])

# possible_moves
v1 = np.array([2, 0, 1])
v2 = np.array([1, 0, 1])
v3 = np.array([1, 1, 1])
v4 = np.array([0, 1, 1])
v5 = np.array([0, 2, 1])
moves = [np.array([2, 0, 1]), np.array([1, 0, 1]), np.array([1, 1, 1]), np.array([0, 1, 1]), np.array([0, 2, 1])]


def values_validation(x):
    """Sprawdza czy wartości tablicy znajdują się we właściwych zakresach"""
    for err in (x < left_limit):
        if err:
            return False
    for err in (x > right_limit):
        if err:
            return False
    return True


def ratio_validation(x):
    """Sprawdza, czy liczba Misjonarzy jest nie mniejsza od Kanibali"""
    if x[0] >= x[1]:
        return True
    else:
        return False


g = Graph()
g.addVertex(tuple(L))   # kolor oryginalnie biały

def m_and_c_problem(g):
    """Funkcja ta ma za zadanie, na podstawie wprowadzonego grafu, dla tych węzłów, które nie były jeszcze sprawdzone
    (białe), zbudować nowe, możliwe opcje przemieszczania się osób. W tym procesie sprawdzane są takie rzeczy jak:

    1) To, z którego brzegu startujemy (dist%2, dist dla każdej kolejnej generacji węzłów się zwiększa o 1, a jego
    parzyste wartości wskazują na lewy brzeg rzeki), aby móc określić czy dodać, czy odjąć wektory "move"

    2) Zakres liczb po dodaniu konkretnych ruchów (metoda values_validation)

    3) Stosunek Misjonarzy do Kanibali (metoda ratio_validation)
    Uwaga! Sprawdzane jest tylko dla brzegu, gdzie w aktualnym momencie nie ma łódki, po drugiej stronie następuje
    proces przeładowywania, a więc przewaga jest dozwolona

    4) To, czy potencjalna nowa konfiguracja wcześniej już nie wystąpiła, wprowadzałoby to dodatkowe niepotrzebne
    komplikacje i obliczenia, ilość potrzebnych ruchów natomiast by się zwiększyła

    Po przejściu wszystkich kryteriów do grafu dodawany jest nowy węzeł, o kolorze white, gotowy do następnego
    wywołania funkcji, stary zaś ma ustawiany kolor black.
    tuple() oraz np.array() wprowadzone w celu przeskakiwania między reprezentacją nadającą się na Id węzła,
    oraz macierzami nadającymi się do działań dodawania i odejmowania

    Funkcja zwraca graf o dodanych nowych węzłach
    """
    verticies = [vert for vert in g]
    for current_vertex in verticies:
        if current_vertex.getColor() == "white":
            dist = current_vertex.getDistance()
            for move in moves:
                if dist % 2:               # 1)
                    new_config = np.array(current_vertex.getId()) + move
                    if values_validation(new_config) and ratio_validation(np.array([3,3,1]) - new_config):  # 2) i 3)
                        if tuple(new_config) not in list(g.getVertices()):                   # 4)
                            add_vertices(tuple(new_config), g, current_vertex, dist + 1, move)
                else:
                    new_config = np.array(current_vertex.getId()) - move
                    if values_validation(new_config) and ratio_validation(new_config):                      # 2) i 3)
                        if tuple(new_config) not in list(g.getVertices()):                     # 4)
                            add_vertices(tuple(new_config), g, current_vertex, dist + 1, move)
            current_vertex.setColor("black")
    # for vert in g:
    #     print(vert)
    return g


def add_vertices(v, g, pred, dist, move):
    """Dodawanie nowych węzłów o odpowiednich atrybutach"""
    made_move = np.array(move)
    if dist%2:
        made_move = - made_move
    g.addVertex(v)
    g.getVertex(v).setPred(pred)
    g.getVertex(v).setDistance(dist)
    g.getVertex(v).setMove(tuple(made_move))


def recursive_solution(g):
    """Funkcja rekurencyjnie wywołująca funkcję m_and_c_problem(), tak długo, aż otrzymamy rozwiązanie."""
    while (0,0,0) not in g:
        a = m_and_c_problem(g)
        g = a
    print("Solution found!")
    nv = g.getVertex((0, 0, 0))
    for i in range(nv.getDistance() + 1):
        print(f"Strony: Lewa {nv.getId()}, Prawa  {tuple(np.array([3,3,1]) - np.array(nv.getId()))},   wykonany ruch {nv.getMove()}")
        nv = nv.getPred()    # print(list(g.getVertices()))


if __name__ == "__main__":
    g = Graph()
    g.addVertex(tuple(L))
    recursive_solution(g)
    print("Po tej stronie gdzie jest łódka (1), może być przewaga kanibali, ponieważ następuje proces przeładunku")
