import sys

class Graph:

    def __init__(self, n):
        self.size = n
        self.adjacencyList = [[] for i in range(n+1)]

    def setEdge(self, u, v):
        self.adjacencyList[u].append(v)
    
    def bipartite(self):
        colorArr = [-1] * (self.size+1)
        colorArr[1] = 1
        queue = []
        queue.append(1)
        while queue:
 
            u = queue.pop()
 
            if u in self.adjacencyList[u]:
                return False;
 
            for v in self.adjacencyList[u]:
        
                if colorArr[v] == -1:
                    colorArr[v] = 1 - colorArr[u]
                    queue.append(v)
                    
                elif colorArr[v] == colorArr[u]:
                    return False
 
        return True


if __name__=="__main__":

    with open(sys.argv[1]) as f:

        lines = f.readlines()
        size = int(lines[0])
        graph = Graph(size)

        for i in range(1, graph.size+1):
            neighbors = [int(v) for v in lines[i].split()]
            for v in neighbors:
                graph.setEdge(i, v)

    print(graph.bipartite())