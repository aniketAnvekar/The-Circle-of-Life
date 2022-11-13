import random
from collections import deque

import MapUtils as mp


class Predator:

    def __init__(self, graph, config, agent_pos, simulation=None):
        if simulation is None:
            while True:
                self.position = random.randrange(0, config["GRAPH_SIZE"])
                if not self.position == agent_pos:
                    break
        else:
            self.position = simulation
        self.graph = graph
        self.last_agent_pos = -1
        self.path = deque()

    def update(self, agent_pos):
        if self.last_agent_pos != agent_pos or len(self.path) == 0:
            self.path = mp.shortest_path_to_goal(self.graph, self.position, agent_pos)
        self.position = self.path.pop()
        self.last_agent_pos = agent_pos

        return -1 if self.position == agent_pos else 0
