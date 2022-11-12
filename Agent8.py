import MapUtils as mp
from random import choice
import numpy as np
import AgentUtils as au


class Agent8:
    def __init__(self, graph, start, config):
        self.position = start
        self.graph = graph
        self.config = config
        self.visited = [0 for _ in self.graph.keys()]

        # prey initialization

        self.prey_q = [1 / (self.config["GRAPH_SIZE"] - 1) for _ in range(self.config["GRAPH_SIZE"])]
        self.prey_q[start] = 0
        self.found_prey = False

        # prey set up transition matrix

        self.prey_P = [[0 for _ in range(self.config["GRAPH_SIZE"])] for _ in range(self.config["GRAPH_SIZE"])]
        for i in graph.keys():
            self.prey_P[i][i] = 1 / (len(self.graph[i]) + 1)
            for j in graph[i]:
                self.prey_P[i][j] = 1 / (len(self.graph[j]) + 1)

        # predator initialization

        self.predator_q = [0 for _ in range(self.config["GRAPH_SIZE"])]
        self.first_run = True

    def update(self, predator, prey):
        if self.first_run:
            # finish initialization...
            self.predator_q[predator.position] = 1
            self.first_run = False

        estimated_predator_position, estimated_prey_position = au.survey_combined(self, predator, prey)

        options = list(filter(lambda x: self.predator_q[x] == self.predator_q[estimated_predator_position], self.graph.keys()))
        if len(options) != 1:
            distances = mp.get_shortest_distances_to_goals(self.graph, self.position, options)
            shortest = min(distances.values())
            estimated_predator_position = choice([i for i in distances.keys() if distances[i] == shortest])

        options = list(filter(lambda x: self.prey_q[x] == self.prey_q[estimated_prey_position], self.graph.keys()))
        if len(options) != 1:
            distances = mp.get_shortest_distances_to_goals(self.graph, estimated_predator_position, options)
            longest = max(distances.values())
            estimated_prey_position = choice([i for i in distances.keys() if distances[i] == longest])

        ret = au.basic_update_agent(self, predator, prey, estimated_predator_position=estimated_predator_position,
                                    estimated_prey_position=estimated_prey_position)
        if ret == 0:
            au.general_move_agent(self)

        return ret


