from random import choice

import AgentUtils as au
import MapUtils as mp


class Agent10A:
    def __init__(self, graph, start, config):
        self.position = start
        self.graph = graph
        self.config = config
        self.total_prey_guess = 0
        self.total_prey_correct = 0
        self.total_pred_guess = 0
        self.total_pred_correct = 0
        self.visited = [0 for _ in self.graph.keys()]

        # initialization

        self.prey_q = [1 / (self.config["GRAPH_SIZE"] - 1) for _ in range(self.config["GRAPH_SIZE"])]
        self.prey_q[start] = 0
        self.found_prey = False

        # set up transition matrix

        self.prey_P = [[0 for _ in range(self.config["GRAPH_SIZE"])] for _ in range(self.config["GRAPH_SIZE"])]
        for i in graph.keys():
            self.prey_P[i][i] = 1 / (len(self.graph[i]) + 1)
            for j in graph[i]:
                self.prey_P[i][j] = 1 / (len(self.graph[j]) + 1)

    def update(self, predator, prey):

        ret = 0
        dist_to_pred = mp.get_shortest_distances_to_goals(self.graph, predator.position, [self.position])[self.position]
        estimated_prey_position = au.pick_most_probable_spot(self, self.prey_q)
        options = list(filter(lambda x: self.prey_q[x] == self.prey_q[estimated_prey_position], self.graph.keys()))
        if len(options) != 1:
            distances = mp.get_shortest_distances_to_goals(self.graph, predator.position, options)
            longest = max(distances.values())
            estimated_prey_position = choice([i for i in distances.keys() if distances[i] == longest])

        if dist_to_pred >= 2:
            au.survey_partial_prey(self, prey)
        else:
            my_neighbors = set(self.graph[self.position])
            pred_neighbors = set(self.graph[predator.position])
            options = my_neighbors - pred_neighbors
            distances = mp.get_shortest_distances_to_goals(self.graph, predator.position, options)
            if len(distances.values()) != 0:
                longest = max(distances.values())
                options = [i for i in distances.keys() if distances[i] == longest]
            distances = mp.get_shortest_distances_to_goals(self.graph, estimated_prey_position, options)
            if len(distances.values()) != 0:
                shortest = min(distances.values())
                options = [i for i in distances.keys() if distances[i] == shortest]
                sorted(options, key=lambda x: self.visited[x])
                self.position = options[0]
            self.visited[self.position] = self.visited[self.position] + 1
            ret = 1 if prey.position == self.position else -1 if predator.position == self.position else 0

        if ret == 0:
            au.general_move_agent(self)
        return ret

