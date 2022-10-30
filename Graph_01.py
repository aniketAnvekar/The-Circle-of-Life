import random
import networkx as nx
import matplotlib.pyplot as plt

class Graph:

    def __init__(self, nodes):
        self.alist = {}
        self.nodes = nodes

    def create(self):
        for i in range(self.nodes-1):
            try:
                self.alist[i].append(i+1)
            except:
                self.alist[i] = [i+1]
            
            self.alist[i+1] = [i]
        self.alist[self.nodes-1] = [0]
        self.alist[self.nodes-1].append(self.nodes-2)
        self.alist[0].append(self.nodes-1)
        self.alist = self.create_additionalEdges()
        print(self.alist)
        return self.alist

    # def getElement(self, arr):
    #     n = len(arr)
    #     i = 0
    #     while i<n:
    #         print(arr)
    #         rand_idx = random.randrange(len(arr))

    #         rand_no = arr[rand_idx]

    #         # print(rand_no)

    #         if(len(self.alist[rand_no])<3):
    #             return rand_no

    #         arr.remove(rand_no)
    #         i += 1
    #     return -1


    def create_additionalEdges(self):

        i = self.nodes
        j = 1
        finished_nodes = 0
        status = [False for i in range(self.nodes)]
        while finished_nodes < self.nodes:
            i = (i + 1) % self.nodes
            if status[i]:
                continue
            else:
                neighbors = [k % self.nodes for k in range(i + 2, i + 6)]
                neighbors.extend([k % self.nodes for k in range(i - 5, i - 1)])
                valid_neighbors = list(filter(lambda x: len(self.alist[x]) < 3, neighbors))
                valid_neighbor_count = len(valid_neighbors)

                if valid_neighbor_count == 0:
                    status[i] = True
                else:
                    random_choice_index = random.randrange(valid_neighbor_count)
                    random_choice = valid_neighbors[random_choice_index]
                    print(j, i, random_choice)
                    j = j + 1
                    self.alist[i].append(random_choice)
                    self.alist[random_choice].append(i)
                    status[i] = len(self.alist[i]) >= 3
                    status[random_choice] = len(self.alist[random_choice]) >= 3
                    finished_nodes = finished_nodes + 1 if status[random_choice] else finished_nodes
                finished_nodes = finished_nodes + 1 if status[i] else finished_nodes

        return self.alist



    def create_edges(self):
        edges = []
        for k,v in self.alist.items():
            for ele in v:
                if (k,ele) not in edges:
                    edges.append((k,ele))
        return edges 

    def visualize(self):
        G = nx.Graph()
        
        for i in range(0, 50):
            G.add_node(i) 
        
        edges = self.create_edges()
        G.add_edges_from(edges)
        return G




       

                # flag = True
                # while flag:
                    



                    # move_decider = random.randint(1,2)
                    # if(move_decider == 1):
                    #     print('Move forward')
                    #     arr = [k%self.nodes for k in range(i+2,i+6)]
                    #     ele = self.getElement(arr)
                    #     if ele and ele!=-1:
                    #         self.alist[i].append(ele)
                    #         self.alist[ele].append(i)
                    #         print(self.alist[i], self.alist[ele])
                    #         flag = False
                    #         break
                    # else:
                    #     print('Move backward') 
                    #     node_list = []
                    #     for k in range(5,1,-1):
                    #         node_list.append((self.nodes+(i-k))%self.nodes)
                    # #    print(i)
                    #     print(node_list)
                    #     ele = self.getElement(node_list)
                    #     if ele and ele!=-1:
                    #         self.alist[i].append(ele)
                    #         self.alist[ele].append(i)
                    #         print(self.alist[i], self.alist[ele])
                    #         flag = False
                    #         break