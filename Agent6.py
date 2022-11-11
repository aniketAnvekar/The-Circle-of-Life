import MapUtils as mp
from random import choice
import numpy as np
import AgentUtils as au

class Agent6:
    def __init__(self, graph, start, config):
        self.position = start
        self.graph = graph
        self.config = config
        self.found_predator = False
        self.visited = [0 for _ in self.graph.keys()]
        self.q = [1/(self.config["GRAPH_SIZE"]) for i in range(self.config["GRAPH_SIZE"])] # vector containing probabilities of where the prey is

    def belief_system(self, predator):


        if not self.found_predator: # agent does know where the predator starts
            self.q = [0 for i in range(self.config["GRAPH_SIZE"])]
            self.q[predator.position] = 1
        else:

            P = self.calculate_transition_probability_matrix()
            self.q = list(np.dot(P, self.q))

            old_agent_pos_prob = self.q[self.position]
            self.q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.q))
            self.q[self.position] = 0

        self.q = au.normalize_probs(self.q)
        au.checkProbSum(sum(self.q))

        max_prob = max(self.q)
        survey_spot = choice([i for i in self.graph.keys() if self.q[i] == max_prob])

        if survey_spot == predator.position:
            self.q = [0 for i in range(self.config["GRAPH_SIZE"])]
            self.q[survey_spot] = 1
            self.found_predator = True
        else:
            old_survey_spot_prob = self.q[survey_spot]
            self.q[survey_spot] = 0
            self.q = list(map(lambda x: x / (1 - old_survey_spot_prob), self.q))

        self.q = au.normalize_probs(self.q)
        au.checkProbSum(sum(self.q))

        max_prob = max(self.q)
        return choice([i for i in self.graph.keys() if self.q[i] == max_prob])

    def calculate_transition_probability_matrix(self):

        shortest_distances = mp.getShortestDistancesToGoals(self.graph, self.position, list(self.graph.keys())[:])
        P = [ [0 for i in range(self.config["GRAPH_SIZE"])] for i in range(self.config["GRAPH_SIZE"])]
        for i in range(len(self.graph.keys())):
            for j in self.graph[i]: # for every neighbor j of i, the probability of moving from j to i
                P[i][j] = 0.4*(1 / len(self.graph[j]))
                distances_from_j_neighbors = {x:shortest_distances[x] for x in self.graph[j]}
                min_dist = min(distances_from_j_neighbors.values())
                shortest_distances_from_j_neighbors = {x:y for (x, y) in distances_from_j_neighbors.items() if y <= min_dist}
                if i in shortest_distances_from_j_neighbors.keys():
                    P[i][j] = P[i][j] + 0.6* (1 / len(shortest_distances_from_j_neighbors.keys()))
        return P


    def update(self, predator, prey):

        estimated_predator_position = self.belief_system(predator)

        return au.advanced_update_agent(self, predator, prey, estimated_pred_position=estimated_predator_position)
