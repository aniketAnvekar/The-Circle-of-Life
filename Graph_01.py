import random
import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, nodes):
        self.graph = {}
        self.nodes = nodes

    def create(self):
        for i in range(self.nodes-1):
            try:
                self.graph[i].append(i+1)
            except:
                self.graph[i] = [i+1]
            
            self.graph[i+1] = [i]
        self.graph[self.nodes-1] = [0]
        self.graph[self.nodes-1].append(self.nodes-2)
        self.graph[0].append(self.nodes-1)
        self.graph = self.create_additionalEdges()
        print(self.graph)
        return self.graph

    def getElement(self, arr):
        n = len(arr)
        i = 0
        while i<n:
            print(arr)
            rand_idx = random.randrange(len(arr))

            rand_no = arr[rand_idx]

            # print(rand_no)

            if(len(self.graph[rand_no])<3):
                return rand_no

            arr.remove(rand_no)
            i += 1
        return -1


    def create_additionalEdges(self):
        for i in range(self.nodes):
            if len(self.graph[i])<3:
                flag = True
                while flag:
                    move_decider = random.randint(1,2)
                    if(move_decider == 1):
                        print('Move forward')
                        arr = [k%self.nodes for k in range(i+2,i+6)]
                        ele = self.getElement(arr)
                        if ele and ele!=-1:
                            self.graph[i].append(ele)
                            self.graph[ele].append(i)
                            print(self.graph[i], self.graph[ele])
                            flag = False
                            break
                    else:
                        print('Move backward') 
                        node_list = []
                        for k in range(5,1,-1):
                            node_list.append((self.nodes+(i-k))%self.nodes)
                    #    print(i)
                        print(node_list)
                        ele = self.getElement(node_list)
                        if ele and ele!=-1:
                            self.graph[i].append(ele)
                            self.graph[ele].append(i)
                            print(self.graph[i], self.graph[ele])
                            flag = False
                            break
        return self.graph

    def create_edges(self):
        edges = []
        for k,v in self.graph.items():
            for ele in v:
                if (k,ele) not in edges:
                    edges.append((k,ele))
        return edges 

    def visualize(self):
        fig = plt.figure()

        fig.set_figheight(15)
        fig.set_figwidth(25)

        G = nx.Graph()
        edges = self.create_edges()
        G.add_edges_from(edges)
        nx.draw_networkx(G)
        plt.show()
                            


g = Graph(50)
g.create()
g.visualize()