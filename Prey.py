import random

class Prey:

    def __init__(self, graph, config, agent_pos):

        while True:
            self.position = random.randrange(0, config["GRAPH_SIZE"])
            if not self.position == agent_pos:
                break

        self.graph = graph

    def update(self, agent_pos):
        # print("Prey Running...")
        old_position = self.position
        neighbors = self.graph[old_position]
        choices = len(neighbors) + 1
        decision = random.randrange(0, choices*100)
        self.position = -1
        for i in range(0, choices - 1):
            if decision < (i+1)*100:
                self.position = neighbors[i]
                break
        if self.position == -1:
            self.position = old_position

        return 1 if self.position == agent_pos else 0

