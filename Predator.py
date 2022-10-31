import queue as q
import MapUtils as mp
import random
from collections import deque

class Predator:

    def __init__(self, graph, config, agent_pos):
        while True:
            self.position = random.randrange(0, config["GRAPH_SIZE"])
            if not self.position == agent_pos:
                break

        self.graph = graph
        self.last_agent_pos = -1
        self.path = deque()

    def update(self, agent_pos):
        # print("Predator Running...")
        if self.last_agent_pos != agent_pos or len(self.path) == 0:
            self.path = mp.shortestPathToGoal(self.graph, self.position, agent_pos)
        # print(self.path, self.position)
        self.position = self.path.pop()
        self.last_agent_pos = agent_pos
        
        return -1 if self.position == agent_pos else 0
            

