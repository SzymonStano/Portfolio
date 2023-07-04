import numpy as np

import time

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
        self.move = 0

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

    # def __str__(self):
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


left_canister = 4
right_canister = 3


# valiation function
def full_validation(left, right):
    """Process of spilling extra water"""
    if left > 4:
        left = 4
    if right > 3:
        right = 3
    return left, right


# Possible action functions
def fill_left(left, right):
    return 4, right


def fill_right(left, right):
    return left, 3


def empty_left(left, right):
    return 0, right


def empty_right(left, right):
    return left, 0


def pour_whole_to_left(left, right):
    return full_validation(left + right, 0)


def pour_whole_to_right(left, right):
    return full_validation(0, right + left)


def pour_to_left_till_full(left, right):
    while left < 4 and right > 0:
        left += 1
        right -= 1
    return left, right


def pour_to_right_till_full(left, right):
    while left > 0 and right < 3:
        left -= 1
        right += 1
    return left, right


func_list = [fill_left, fill_right, empty_left, empty_right, pour_whole_to_left,
             pour_whole_to_right, pour_to_left_till_full, pour_to_right_till_full]


def two_canisters_problem(g):
    """Funkcja identycznie działającą do tej z zad 6. Bazujemy tym razem na innych ruchach oraz systemie dwóch liczb"""
    vertices = [vert for vert in g]
    for current_vertex in vertices:
        if current_vertex.getColor() == "white":
            dist = current_vertex.getDistance()
            for func in func_list:
                new_config = func(current_vertex.getId()[0], current_vertex.getId()[1])
                if new_config not in g.getVertices():
                    add_vertices(new_config, g, current_vertex, dist + 1, func)
            current_vertex.setColor("black")
    # for vert in g:
    #     print(vert)
    # print(f"Round has ended")
    # print("")
    return g


def add_vertices(v, g, pred, dist, move):
    """Dodawanie nowych węzłów o odpowiednich atrybutach"""
    g.addVertex(v)
    g.getVertex(v).setPred(pred)
    g.getVertex(v).setDistance(dist)
    g.getVertex(v).setMove(move.__name__)


def recursive_solution(g):
    """Funkcja rekurencyjnie wywołująca funkcję two_canisters_problem(), tak długo, aż otrzymamy rozwiązanie."""
    # while (0, 2) and (2, 0) not in g:
    while (0, 2) not in g and (2, 0) not in g:
        a = two_canisters_problem(g)
        g = a
    print("Solution found!")
    if (0, 2) in g:
        nv = g.getVertex((0, 2))
    else:
        nv = g.getVertex((2, 0))
    for i in range(nv.getDistance() + 1):
        print(f"Stan: {nv.getId()}   wykonany ruch: {nv.getMove()}")
        nv = nv.getPred()    # print(list(g.getVertices()))


if __name__ == "__main__":
    g = Graph()
    g.addVertex((0, 0))
    t = time.perf_counter()
    recursive_solution(g)
    t2 = time.perf_counter()
    print(t2-t)

