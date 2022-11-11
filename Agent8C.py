import random

import MapUtils as mp
from random import choice
import numpy as np
import AgentUtils as au

class Agent7C:

    # This is Agent 7 with a defective survey drone


    def __init__(self, graph, start, config):
        self.position = start
        self.graph = graph
        self.config = config
        self.prey_q = [1/(self.config["GRAPH_SIZE"]) for i in range(self.config["GRAPH_SIZE"])] # vector containing probabilities of where the prey is
        self.predator_q = []
        self.visited = [0 for _ in self.graph.keys()]
        self.found_prey = False
        self.prey_P = [ [0 for i in range(self.config["GRAPH_SIZE"])] for i in range(self.config["GRAPH_SIZE"])] # transition probabilities from j to i
        for i in graph.keys():
            self.prey_P[i][i] = 1/(len(self.graph[i]) + 1)
            for j in graph[i]:
                self.prey_P[i][j] = 1/(len(self.graph[j]) + 1)
        self.found_predator = False
        self.predator_q = [1/(self.config["GRAPH_SIZE"]) for i in range(self.config["GRAPH_SIZE"])] # vector containing probabilities of where the prey is

    def belief_system(self, predator, prey):

        if not self.found_predator: # agent does know where the predator starts
            # print("Running initialization...")
            self.predator_q = [0 for i in range(self.config["GRAPH_SIZE"])]
            self.predator_q[predator.position] = 1
            self.found_predator = True
        else:

            predator_P = self.calculate_transition_probability_matrix()
            self.predator_q = list(np.dot(predator_P, self.predator_q))

            old_agent_pos_prob = self.predator_q[self.position]
            self.predator_q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.predator_q))
            self.predator_q[self.position] = 0

        if not self.found_prey:
            # initialization
            self.prey_q = [1/(self.config["GRAPH_SIZE"] - 1) for i in range(self.config["GRAPH_SIZE"])]
            self.prey_q[self.position] = 0
        else:
            # calculate the probabilities of the nodes at each path depending on when the prey was last seen
            self.prey_q = list(np.dot(self.prey_P, self.prey_q))

            old_agent_pos_prob = self.prey_q[self.position]
            self.prey_q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.prey_q))
            self.prey_q[self.position] = 0

        self.prey_q = au.normalize_probs(self.prey_q)
        au.check_prob_sum(sum(self.prey_q))
        self.predator_q = au.normalize_probs(self.predator_q)
        au.check_prob_sum(sum(self.predator_q))

        max_predator_prob = max(self.predator_q)
        survey_spot = 0
        defective = random.randrange(100) < 10
        if max_predator_prob != 1:
            survey_spot = choice([i for i in self.graph.keys() if self.predator_q[i] == max_predator_prob])
        else:
            max_prey_prob = max(self.prey_q)
            survey_spot = choice([i for i in self.graph.keys() if self.prey_q[i] == max_prey_prob])

        if defective:
            old_survey_spot_prob = self.prey_q[survey_spot]
            self.prey_q[survey_spot] = 0.1*old_survey_spot_prob
            self.prey_q = list(map(lambda x: x / (0.1*old_survey_spot_prob + (1 - old_survey_spot_prob)), self.prey_q))
            old_survey_spot_prob = self.predator_q[survey_spot]
            if old_survey_spot_prob != 1:
                self.predator_q[survey_spot] = 0.1*old_survey_spot_prob
                self.predator_q = list(map(lambda x: x / (0.1*old_survey_spot_prob + (1 - old_survey_spot_prob)), self.predator_q))

        else:
            if survey_spot == prey.position:
                # P(prey in X | survey drone scans prey at S) = P(survey drone scans prey at S | prey in X)P(prey in X) / P(survey drone scans prey at S)
                # P(survey drone scans prey at S) = P(survey drone scans prey at S | prey in S)P(prey in S) + P(survey drone scans prey at S | prey not in S)P(prey not in S)
                # P(survey drone scans prey at S | prey in X) = 0 if X != S, 0.9 if X == S
                # if X != S
                # P(prey in X | survey drone scans prey at S) = 0*P(prey in X) / (0.9*P(prey in S) + 0*P(prey not in S)) = 0
                # if X == S
                # P(prey in X | survey drone scans prey at S) =  0.9*P(prey in S) / (0.9*P(prey in S) + 0*P(prey not in S)) = 1

                self.prey_q = [0 for i in range(self.config["GRAPH_SIZE"])]
                self.prey_q[survey_spot] = 1
                self.found_prey = True
            else:
                # P(prey in X | survey drone doesn't scan prey at S) = P(survey drone doesn't scan prey at S | prey in X)P(prey in X) / P(survey drone doesn't scan prey at S)
                # P(survey drone doesn't scan prey at S | prey in X) = 1 if X != S, 0.1 if X == S
                # P(survey drone doesn't scan prey at S) = P(survey drone doesn't scan prey at S | prey in S)P(prey in S) + P(survey drone doesn't scan prey at S | prey not in S)P(prey not in S)
                # = 0.1*P(prey in S) + 1*P(prey not in S)
                # for X != S
                # P(prey in X | survey drone doesn't scan prey at S) = 1*P(prey in X) / (0.1*P(prey in S) + 1*P(prey not in S))
                # for X == S
                # P(prey in X | survey drone doesn't scan prey at S) = 0.1*P(prey in S) / (0.1*P(prey in S) + 1*P(prey not in S))

                old_survey_spot_prob = self.prey_q[survey_spot]
                self.prey_q[survey_spot] = 0.1*old_survey_spot_prob
                self.prey_q = list(map(lambda x: x / (0.1*old_survey_spot_prob + (1 - old_survey_spot_prob)), self.prey_q))

            # print("Survey Spot: " + str(survey_spot))

            if survey_spot == predator.position:
                # P(pred in X | survey drone scans pred at S) = P(survey drone scans pred at S | pred in X)P(pred in X) / P(survey drone scans pred at S)
                # P(survey drone scans pred at S) = P(survey drone scans pred at S | pred in S)P(pred in S) + P(survey drone scans pred at S | pred not in S)P(pred not in S)
                # P(survey drone scans pred at S | pred in X) = 0 if X != S, 0.9 if X == S
                # if X != S
                # P(pred in X | survey drone scans pred at S) = 0*P(pred in X) / (0.9*P(pred in S) + 0*P(pred not in S)) = 0
                # if X == S
                # P(pred in X | survey drone scans pred at S) =  0.9*P(pred in S) / (0.9*P(pred in S) + 0*P(pred not in S)) = 1
                old_survey_spot_prob = self.predator_q[survey_spot]
                self.predator_q = [0 for i in range(self.config["GRAPH_SIZE"])]
                self.predator_q[survey_spot] = 1
                self.found_predator = True
            # print("Predator Found!")
            else:
                # P(pred in X | survey drone doesn't scan pred at S) = P(survey drone doesn't scan pred at S | pred in X)P(pred in X) / P(survey drone doesn't scan pred at S)
                # P(survey drone doesn't scan pred at S | pred in X) = 1 if X != S, 0.1 if X == S
                # P(survey drone doesn't scan pred at S) = P(survey drone doesn't scan pred at S | pred in S)P(pred in S) + P(survey drone doesn't scan pred at S | pred not in S)P(pred not in S)
                # = 0.1*P(pred in S) + 1*P(pred not in S)
                # for X != S
                # P(pred in X | survey drone doesn't scan pred at S) = 1*P(pred in X) / (0.1*P(pred in S) + 1*P(pred not in S))
                # for X == S
                # P(pred in X | survey drone doesn't scan pred at S) = 0.1*P(pred in S) / (0.1*P(pred in S) + 1*P(pred not in S))
                old_survey_spot_prob = self.predator_q[survey_spot]
                self.predator_q[survey_spot] = 0.1*old_survey_spot_prob
                self.predator_q = list(map(lambda x: x / (0.1*old_survey_spot_prob + (1 - old_survey_spot_prob)), self.predator_q))

        self.prey_q = au.normalize_probs(self.prey_q)
        au.check_prob_sum(sum(self.prey_q))
        self.predator_q = au.normalize_probs(self.predator_q)
        au.check_prob_sum(sum(self.predator_q))

        max_prey_prob = max(self.prey_q)
        max_predator_prob = max(self.predator_q)
        return choice([i for i in self.graph.keys() if self.predator_q[i] == max_predator_prob]), choice([i for i in self.graph.keys() if self.prey_q[i] == max_prey_prob])

    def calculate_transition_probability_matrix(self):

        shortest_distances = mp.get_shortest_distances_to_goals(self.graph, self.position, list(self.graph.keys())[:])
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

        estimated_predator_position, estimated_prey_position = self.belief_system(predator, prey)

        return au.advanced_update_agent(self, predator, prey, estimated_predator_position=estimated_predator_position, estimated_prey_position=estimated_prey_position)
