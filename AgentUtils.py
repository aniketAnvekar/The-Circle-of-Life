import random

import Agent7
import Agent7B
import Agent7C
import Agent8
import Agent8B
import Agent8C
import MapUtils as mp
from random import choice
import Predator as pr
import numpy as np
import Agent1
import Agent2
import Agent3
import Agent4
import Agent5
import Agent6


prey_type = (Agent3.Agent3, Agent4.Agent4, Agent7.Agent7, Agent8.Agent8, Agent7B.Agent7B, Agent8B.Agent8B, Agent7C.Agent7C, Agent8C.Agent8C)
pred_type = (Agent5.Agent5, Agent6.Agent6, Agent7.Agent7, Agent8.Agent8, Agent7B.Agent7B, Agent8B.Agent8B, Agent7C.Agent7C, Agent8C.Agent8C)


def basic_update_agent(agent, predator, prey, estimated_predator_position=None, estimated_prey_position=None):
    neighbors_and_self = agent.graph[agent.position][:]
    neighbors_and_self.append(agent.position)

    if estimated_predator_position is None:
        predator_distances = mp.get_shortest_distances_to_goals(agent.graph, predator.position, neighbors_and_self[:])
    else:
        predator_distances = mp.get_shortest_distances_to_goals(agent.graph, estimated_predator_position,
                                                                neighbors_and_self[:])

    if estimated_prey_position is None:
        prey_distances = mp.get_shortest_distances_to_goals(agent.graph, prey.position, neighbors_and_self[:])
    else:
        prey_distances = mp.get_shortest_distances_to_goals(agent.graph, estimated_prey_position, neighbors_and_self[:])

    cur_dist_pred = predator_distances[agent.position]
    cur_dist_prey = prey_distances[agent.position]

    closer_to_prey = set(
        [x for x in prey_distances.keys() if prey_distances[x] < cur_dist_prey and x != agent.position])
    same_to_prey = set([x for x in prey_distances.keys() if prey_distances[x] == cur_dist_prey and x != agent.position])
    same_to_pred = set(
        [x for x in predator_distances.keys() if predator_distances[x] == cur_dist_pred and x != agent.position])
    far_from_pred = set(
        [x for x in predator_distances.keys() if predator_distances[x] > cur_dist_pred and x != agent.position])

    closer_to_prey_and_further_from_pred = closer_to_prey.intersection(far_from_pred)
    closer_to_prey_and_same_from_pred = closer_to_prey.intersection(same_to_pred)
    same_to_prey_and_further_from_pred = same_to_prey.intersection(far_from_pred)
    same_to_prey_and_same_from_pred = same_to_prey.intersection(same_to_pred)

    if len(closer_to_prey_and_further_from_pred) != 0:
        agent.position = choice(list(closer_to_prey_and_further_from_pred))
    elif len(closer_to_prey_and_same_from_pred) != 0:
        agent.position = choice(list(closer_to_prey_and_same_from_pred))
    elif len(same_to_prey_and_further_from_pred) != 0:
        agent.position = choice(list(same_to_prey_and_further_from_pred))
    elif len(same_to_prey_and_same_from_pred) != 0:
        agent.position = choice(list(same_to_prey_and_same_from_pred))
    elif len(far_from_pred) != 0:
        agent.position = choice(list(far_from_pred))
    elif len(same_to_pred) != 0:
        agent.position = choice(list(same_to_pred))

    return 1 if agent.position == prey.position else -1 if agent.position == predator.position else 0


def advanced_update_agent(agent, predator, prey, estimated_predator_position=None, estimated_prey_position=None):
    min_neighbor = -1
    min_dist = 1000
    for neighbor in agent.graph[agent.position]:

        sim_predator = pr.Predator(agent.graph, agent.config, agent.position,
                                   simulation=predator.position if estimated_predator_position is None else estimated_predator_position)
        dist = mp.recursive_search(agent, agent.config["DEPTH"], neighbor, sim_predator,
                                   prey.position if estimated_prey_position is None else estimated_prey_position, set())

        if dist is not None:
            if dist < min_dist or (dist == min_dist and agent.visited[neighbor] <= agent.visited[min_neighbor]):
                min_dist = dist
                min_neighbor = neighbor

    if min_neighbor != -1:
        agent.position = min_neighbor
    agent.visited[agent.position] = agent.visited[agent.position] + 1

    return 1 if agent.position == prey.position else -1 if agent.position == predator.position else 0


def normalize_probs(vector):
    s = sum(vector)
    vector = list(map(lambda x: x / s, vector))
    return vector


def check_prob_sum(su):
    if abs(1 - su) < 0.00000000000001:  # 0.000000000000001
        return
    print("BELIEF SYSTEM FAULTY: " + str(su))
    exit()


def normalize_and_check(vector):
    normalize_probs(vector)
    check_prob_sum(sum(vector))


def pick_most_probable_spot(agent, vector):
    max_prob = max(vector)
    return choice([i for i in agent.graph.keys() if vector[i] == max_prob])


#################### SURVEY FUNCTIONS ############################

def survey_defective_drone(agent, predator, prey, defective=None):
    survey_spot = pick_most_probable_spot(agent, agent.predator_q if max(agent.predator_q) != 1 else agent.prey_q)
    error = random.randrange(100) < 10
    if error:  # false negative
        agent.prey_q = survey_negative_response(agent.prey_q, survey_spot, defective)
        if agent.predator_q[survey_spot] != 1:
            agent.predator_q = survey_negative_response(agent.predator_q, survey_spot, defective)
        return pick_most_probable_spot(agent, agent.predator_q), pick_most_probable_spot(agent, agent.prey_q)
    return survey_partial_pred(agent, predator, survey_spot=survey_spot, defective=defective), survey_partial_prey(agent, prey,
                                                                                                  survey_spot=survey_spot, defective=defective)


def survey_combined(agent, predator, prey):
    survey_spot = pick_most_probable_spot(agent, agent.predator_q if max(agent.predator_q) != 1 else agent.prey_q)
    return survey_partial_pred(agent, predator, survey_spot=survey_spot), survey_partial_prey(agent, prey,
                                                                                              survey_spot=survey_spot)

def survey_partial_pred(agent, predator, survey_spot=None, defective=None):
    survey_spot = pick_most_probable_spot(agent, agent.predator_q) if survey_spot is None else survey_spot
    if survey_spot == predator.position:
        agent.predator_q = [0 for _ in range(agent.config["GRAPH_SIZE"])]
        agent.predator_q[survey_spot] = 1
    else:
        agent.predator_q = survey_negative_response(agent.predator_q, survey_spot, defective)
    normalize_and_check(agent.predator_q)
    return pick_most_probable_spot(agent, agent.predator_q)


def survey_partial_prey(agent, prey, survey_spot=None, defective=None):
    survey_spot = pick_most_probable_spot(agent, agent.prey_q) if survey_spot is None else survey_spot
    if survey_spot == prey.position:
        agent.prey_q = [0 for _ in range(agent.config["GRAPH_SIZE"])]
        agent.prey_q[survey_spot] = 1
        agent.found_prey = True
    else:
        agent.prey_q = survey_negative_response(agent.prey_q, survey_spot, defective)
    normalize_and_check(agent.prey_q)
    return pick_most_probable_spot(agent, agent.prey_q)


def survey_negative_response(vector, survey_spot, defective):
    old_survey_spot_prob = vector[survey_spot]
    vector[survey_spot] = 0 if defective is None else 0.1*old_survey_spot_prob
    return list(map(lambda x: x / (0.1*old_survey_spot_prob + (1 - old_survey_spot_prob)), vector)) if defective is not None \
        else list(map(lambda x: x / (1 - old_survey_spot_prob), vector))


################################################################
#################### MOVE AGENT FUNCTIONS ######################

def general_move_agent(agent):
    if isinstance(agent, pred_type):
        old_agent_pos_prob = agent.predator_q[agent.position]
        agent.predator_q = list(map(lambda x: x / (1 - old_agent_pos_prob), agent.predator_q))
        agent.predator_q[agent.position] = 0
        normalize_and_check(agent.predator_q)
    if isinstance(agent, prey_type):
        if not agent.found_prey:
            agent.prey_q = [1 / (agent.config["GRAPH_SIZE"] - 1) for _ in range(agent.config["GRAPH_SIZE"])]
        else:
            old_agent_pos_prob = agent.prey_q[agent.position]
            agent.prey_q = list(map(lambda x: x / (1 - old_agent_pos_prob), agent.prey_q))
        agent.prey_q[agent.position] = 0
        normalize_and_check(agent.prey_q)


################################################################
#################### MOVE PIECE FUNCTIONS ######################


def belief_system_move_pieces(agent):
    if isinstance(agent, (Agent1.Agent1, Agent2.Agent2)):
        return
    if isinstance(agent, prey_type) and agent.found_prey:
        agent.prey_q = list(np.dot(agent.prey_P, agent.prey_q))
        agent.prey_q = normalize_probs(agent.prey_q)
        check_prob_sum(sum(agent.prey_q))
    if isinstance(agent, pred_type):
        P = calculate_transition_probability_matrix(agent)
        agent.predator_q = list(np.dot(P, agent.predator_q))
        agent.predator_q = normalize_probs(agent.predator_q)
        check_prob_sum(sum(agent.predator_q))


def calculate_transition_probability_matrix(agent):
    shortest_distances = mp.get_shortest_distances_to_goals(agent.graph, agent.position, list(agent.graph.keys())[:])
    P = [[0 for _ in range(agent.config["GRAPH_SIZE"])] for _ in range(agent.config["GRAPH_SIZE"])]
    for i in range(len(agent.graph.keys())):
        for j in agent.graph[i]:  # for every neighbor j of i, the probability of moving from j to i
            P[i][j] = 0.4 * (1 / len(agent.graph[j]))
            distances_from_j_neighbors = {x: shortest_distances[x] for x in agent.graph[j]}
            min_dist = min(distances_from_j_neighbors.values())
            shortest_distances_from_j_neighbors = {x: y for (x, y) in distances_from_j_neighbors.items() if
                                                   y <= min_dist}
            if i in shortest_distances_from_j_neighbors.keys():
                P[i][j] = P[i][j] + 0.6 * (1 / len(shortest_distances_from_j_neighbors.keys()))
    return P


####################### OLD CODE #######################################################################################
# AGENT 4 OLD CODE #####################################################################################################
# def belief_system(self, prey):
#
#     if not self.found_prey:
#         self.q = [1 / (self.config["GRAPH_SIZE"] - 1) for _ in range(self.config["GRAPH_SIZE"])]
#         self.q[self.position] = 0
#     else:
#         self.q = list(np.dot(self.P, self.q))
#
#         old_agent_pos_prob = self.q[self.position]
#         self.q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.q))
#         self.q[self.position] = 0
#
#     self.q = au.normalize_probs(self.q)
#     au.check_prob_sum(sum(self.q))
#
#     max_prob = max(self.q)
#     survey_spot = choice([i for i in self.graph.keys() if self.q[i] == max_prob])
#
#     if survey_spot == prey.position:
#         self.q = [0 for _ in range(self.config["GRAPH_SIZE"])]
#         self.q[survey_spot] = 1
#         self.found_prey = True
#
#     else:
#         old_survey_spot_prob = self.q[survey_spot]
#         self.q[survey_spot] = 0
#         self.q = list(map(lambda x: x / (1 - old_survey_spot_prob), self.q))
#
#     self.q = au.normalize_probs(self.q)
#     au.check_prob_sum(sum(self.q))
#
#     max_prob = max(self.q)
#     return choice([i for i in self.graph.keys() if self.q[i] == max_prob])
# AGENT 4 OLD CODE #####################################################################################################
# AGENT 5 OLD CODE #####################################################################################################
#   # def belief_system(self, predator):
#
#     if not self.found_predator:  # agent does know where the predator starts
#         self.q = [0 for i in range(self.config["GRAPH_SIZE"])]
#         self.q[predator.position] = 1
#     else:
#
#         P = self.calculate_transition_probability_matrix()
#         self.q = list(np.dot(P, self.q))
#
#         old_agent_pos_prob = self.q[self.position]
#         self.q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.q))
#         self.q[self.position] = 0
#
#     self.q = au.normalize_probs(self.q)
#     au.check_prob_sum(sum(self.q))
#
#     max_prob = max(self.q)
#     survey_spot = choice([i for i in self.graph.keys() if self.q[i] == max_prob])
#
#     if survey_spot == predator.position:
#         self.q = [0 for i in range(self.config["GRAPH_SIZE"])]
#         self.q[survey_spot] = 1
#         self.found_predator = True
#     else:
#         old_survey_spot_prob = self.q[survey_spot]
#         self.q[survey_spot] = 0
#         self.q = list(map(lambda x: x / (1 - old_survey_spot_prob), self.q))
#
#     # print("Total Belief Sum: " + str(sum(self.q)))
#     self.q = au.normalize_probs(self.q)
#     au.check_prob_sum(sum(self.q))
#
#     max_prob = max(self.q)
#     return choice([i for i in self.graph.keys() if self.q[i] == max_prob])


# def belief_system_survey(self, predator):
#     max_prob = max(self.q)
#     survey_spot = choice([i for i in self.graph.keys() if self.q[i] == max_prob])
#
#     if survey_spot == predator.position:
#         self.q = [0 for _ in range(self.config["GRAPH_SIZE"])]
#         self.q[survey_spot] = 1
#         self.found_predator = True
#     else:
#         old_survey_spot_prob = self.q[survey_spot]
#         self.q[survey_spot] = 0
#         self.q = list(map(lambda x: x / (1 - old_survey_spot_prob), self.q))
#
#     self.q = au.normalize_probs(self.q)
#     au.check_prob_sum(sum(self.q))
#
#     max_prob = max(self.q)
#     return choice([i for i in self.graph.keys() if self.q[i] == max_prob])
#
# def belief_system_move_agent(self, predator):
#     if not self.found_predator:  # agent does know where the predator starts
#         self.q = [0 for _ in range(self.config["GRAPH_SIZE"])]
#         self.q[predator.position] = 1
#     else:
#
#         old_agent_pos_prob = self.q[self.position]
#         self.q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.q))
#         self.q[self.position] = 0
#
# def believe_system_move_pieces(self):
#     P = self.calculate_transition_probability_matrix()
#     self.q = list(np.dot(P, self.q))
#
# def calculate_transition_probability_matrix(self):
#
#     shortest_distances = mp.get_shortest_distances_to_goals(self.graph, self.position, list(self.graph.keys())[:])
#     P = [[0 for _ in range(self.config["GRAPH_SIZE"])] for _ in range(self.config["GRAPH_SIZE"])]
#     for i in range(len(self.graph.keys())):
#         for j in self.graph[i]:  # for every neighbor j of i, the probability of moving from j to i
#             P[i][j] = 0.4 * (1 / len(self.graph[j]))
#             distances_from_j_neighbors = {x: shortest_distances[x] for x in self.graph[j]}
#             min_dist = min(distances_from_j_neighbors.values())
#             shortest_distances_from_j_neighbors = {x: y for (x, y) in distances_from_j_neighbors.items() if
#                                                    y <= min_dist}
#             if i in shortest_distances_from_j_neighbors.keys():
#                 P[i][j] = P[i][j] + 0.6 * (1 / len(shortest_distances_from_j_neighbors.keys()))
#     return P
# AGENT 5 OLD CODE #####################################################################################################
# AGENT 6 OLD CODE #####################################################################################################
# def belief_system(self, predator):
#
#
#     if not self.found_predator: # agent does know where the predator starts
#         self.q = [0 for i in range(self.config["GRAPH_SIZE"])]
#         self.q[predator.position] = 1
#     else:
#
#         P = self.calculate_transition_probability_matrix()
#         self.q = list(np.dot(P, self.q))
#
#         old_agent_pos_prob = self.q[self.position]
#         self.q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.q))
#         self.q[self.position] = 0
#
#     self.q = au.normalize_probs(self.q)
#     au.check_prob_sum(sum(self.q))
#
#     max_prob = max(self.q)
#     survey_spot = choice([i for i in self.graph.keys() if self.q[i] == max_prob])
#
#     if survey_spot == predator.position:
#         self.q = [0 for i in range(self.config["GRAPH_SIZE"])]
#         self.q[survey_spot] = 1
#         self.found_predator = True
#     else:
#         old_survey_spot_prob = self.q[survey_spot]
#         self.q[survey_spot] = 0
#         self.q = list(map(lambda x: x / (1 - old_survey_spot_prob), self.q))
#
#     self.q = au.normalize_probs(self.q)
#     au.check_prob_sum(sum(self.q))
#
#     max_prob = max(self.q)
#     return choice([i for i in self.graph.keys() if self.q[i] == max_prob])
#
# def calculate_transition_probability_matrix(self):
#
#     shortest_distances = mp.get_shortest_distances_to_goals(self.graph, self.position, list(self.graph.keys())[:])
#     P = [ [0 for i in range(self.config["GRAPH_SIZE"])] for i in range(self.config["GRAPH_SIZE"])]
#     for i in range(len(self.graph.keys())):
#         for j in self.graph[i]: # for every neighbor j of i, the probability of moving from j to i
#             P[i][j] = 0.4*(1 / len(self.graph[j]))
#             distances_from_j_neighbors = {x:shortest_distances[x] for x in self.graph[j]}
#             min_dist = min(distances_from_j_neighbors.values())
#             shortest_distances_from_j_neighbors = {x:y for (x, y) in distances_from_j_neighbors.items() if y <= min_dist}
#             if i in shortest_distances_from_j_neighbors.keys():
#                 P[i][j] = P[i][j] + 0.6* (1 / len(shortest_distances_from_j_neighbors.keys()))
#     return P
#
# AGENT 6 OLD CODE #####################################################################################################
# AGENT 7 OLD CODE #####################################################################################################
# def checkProbSum(self, su):
# 	if abs(1 - su) < 0.000000001:  # 0.000000000000001
# 		return
# 	print("BELIEF SYSTEM FAULTY")
# 	exit()
#
# def belief_system(self, predator, prey):
#
# 	if not self.found_predator:  # agent does know where the predator starts
# 		# print("Running initialization...")
# 		self.predator_q = [0 for i in range(self.config["GRAPH_SIZE"])]
# 		self.predator_q[predator.position] = 1
# 		self.found_predator = True
# 	else:
#
# 		predator_P = self.calculate_transition_probability_matrix()
# 		self.predator_q = list(np.dot(predator_P, self.predator_q))
#
# 		old_agent_pos_prob = self.predator_q[self.position]
# 		self.predator_q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.predator_q))
# 		self.predator_q[self.position] = 0
#
# 	if not self.found_prey:
# 		# initialization
# 		self.prey_q = [1 / (self.config["GRAPH_SIZE"] - 1) for i in range(self.config["GRAPH_SIZE"])]
# 		self.prey_q[self.position] = 0
# 	else:
# 		# calculate the probabilities of the nodes at each path depending on when the prey was last seen
# 		self.prey_q = list(np.dot(self.prey_P, self.prey_q))
#
# 		old_agent_pos_prob = self.prey_q[self.position]
# 		self.prey_q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.prey_q))
# 		self.prey_q[self.position] = 0
#
# 	self.prey_q = au.normalize_probs(self.prey_q)
# 	au.check_prob_sum(sum(self.prey_q))
# 	self.predator_q = au.normalize_probs(self.predator_q)
# 	au.check_prob_sum(sum(self.predator_q))
#
# 	max_predator_prob = max(self.predator_q)
# 	survey_spot = 0
# 	if max_predator_prob != 1:
# 		survey_spot = choice([i for i in self.graph.keys() if self.predator_q[i] == max_predator_prob])
# 	else:
# 		max_prey_prob = max(self.prey_q)
# 		survey_spot = choice([i for i in self.graph.keys() if self.prey_q[i] == max_prey_prob])
#
# 	if survey_spot == prey.position:
# 		self.prey_q = [0 for i in range(self.config["GRAPH_SIZE"])]
# 		self.prey_q[survey_spot] = 1
# 		self.found_prey = True
# 	else:
# 		old_survey_spot_prob = self.prey_q[survey_spot]
# 		self.prey_q[survey_spot] = 0
# 		self.prey_q = list(map(lambda x: x / (1 - old_survey_spot_prob), self.prey_q))
#
# 	if survey_spot == predator.position:
# 		self.predator_q = [0 for i in range(self.config["GRAPH_SIZE"])]
# 		self.predator_q[survey_spot] = 1
# 		self.found_predator = True
# 	else:
# 		old_survey_spot_prob = self.predator_q[survey_spot]
# 		self.predator_q[survey_spot] = 0
# 		self.predator_q = list(map(lambda x: x / (1 - old_survey_spot_prob), self.predator_q))
#
# 	self.prey_q = au.normalize_probs(self.prey_q)
# 	au.check_prob_sum(sum(self.prey_q))
# 	self.predator_q = au.normalize_probs(self.predator_q)
# 	au.check_prob_sum(sum(self.predator_q))
#
# 	max_prey_prob = max(self.prey_q)
# 	max_predator_prob = max(self.predator_q)
# 	return choice([i for i in self.graph.keys() if self.predator_q[i] == max_predator_prob]), choice(
# 		[i for i in self.graph.keys() if self.prey_q[i] == max_prey_prob])
#
# def calculate_transition_probability_matrix(self):
#
# 	shortest_distances = mp.get_shortest_distances_to_goals(self.graph, self.position, list(self.graph.keys())[:])
# 	P = [[0 for i in range(self.config["GRAPH_SIZE"])] for i in range(self.config["GRAPH_SIZE"])]
# 	for i in range(len(self.graph.keys())):
# 		for j in self.graph[i]:  # for every neighbor j of i, the probability of moving from j to i
# 			P[i][j] = 0.4 * (1 / len(self.graph[j]))
# 			distances_from_j_neighbors = {x: shortest_distances[x] for x in self.graph[j]}
# 			min_dist = min(distances_from_j_neighbors.values())
# 			shortest_distances_from_j_neighbors = {x: y for (x, y) in distances_from_j_neighbors.items() if
# 												   y <= min_dist}
# 			if i in shortest_distances_from_j_neighbors.keys():
# 				P[i][j] = P[i][j] + 0.6 * (1 / len(shortest_distances_from_j_neighbors.keys()))
# 	return P
# AGENT 7 OLD CODE #####################################################################################################
# AGENT 8 OLD CODE #####################################################################################################
#     def checkProbSum(self, su):
#         if abs(1 - su) < 0.000000001:  # 0.000000000000001
#             return
#         print("BELIEF SYSTEM FAULTY")
#         exit()
#
#     def belief_system(self, predator, prey):
#
#         if not self.found_predator:  # agent does know where the predator starts
#             # print("Running initialization...")
#             self.predator_q = [0 for i in range(self.config["GRAPH_SIZE"])]
#             self.predator_q[predator.position] = 1
#             self.found_predator = True
#         else:
#
#             predator_P = self.calculate_transition_probability_matrix()
#             self.predator_q = list(np.dot(predator_P, self.predator_q))
#
#             old_agent_pos_prob = self.predator_q[self.position]
#             self.predator_q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.predator_q))
#             self.predator_q[self.position] = 0
#
#         if not self.found_prey:
#             # initialization
#             self.prey_q = [1 / (self.config["GRAPH_SIZE"] - 1) for i in range(self.config["GRAPH_SIZE"])]
#             self.prey_q[self.position] = 0
#         else:
#             # calculate the probabilities of the nodes at each path depending on when the prey was last seen
#             self.prey_q = list(np.dot(self.prey_P, self.prey_q))
#
#             old_agent_pos_prob = self.prey_q[self.position]
#             self.prey_q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.prey_q))
#             self.prey_q[self.position] = 0
#
#         self.prey_q = au.normalize_probs(self.prey_q)
#         au.check_prob_sum(sum(self.prey_q))
#         self.predator_q = au.normalize_probs(self.predator_q)
#         au.check_prob_sum(sum(self.predator_q))
#
#         max_predator_prob = max(self.predator_q)
#         survey_spot = 0
#         if max_predator_prob != 1:
#             survey_spot = choice([i for i in self.graph.keys() if self.predator_q[i] == max_predator_prob])
#         else:
#             max_prey_prob = max(self.prey_q)
#             survey_spot = choice([i for i in self.graph.keys() if self.prey_q[i] == max_prey_prob])
#
#         if survey_spot == prey.position:
#             self.prey_q = [0 for i in range(self.config["GRAPH_SIZE"])]
#             self.prey_q[survey_spot] = 1
#             self.found_prey = True
#         else:
#             old_survey_spot_prob = self.prey_q[survey_spot]
#             self.prey_q[survey_spot] = 0
#             self.prey_q = list(map(lambda x: x / (1 - old_survey_spot_prob), self.prey_q))
#
#         if survey_spot == predator.position:
#             self.predator_q = [0 for i in range(self.config["GRAPH_SIZE"])]
#             self.predator_q[survey_spot] = 1
#             self.found_predator = True
#         else:
#             old_survey_spot_prob = self.predator_q[survey_spot]
#             self.predator_q[survey_spot] = 0
#             self.predator_q = list(map(lambda x: x / (1 - old_survey_spot_prob), self.predator_q))
#
#         self.prey_q = au.normalize_probs(self.prey_q)
#         au.check_prob_sum(sum(self.prey_q))
#         self.predator_q = au.normalize_probs(self.predator_q)
#         au.check_prob_sum(sum(self.predator_q))
#
#         max_prey_prob = max(self.prey_q)
#         max_predator_prob = max(self.predator_q)
#         return choice([i for i in self.graph.keys() if self.predator_q[i] == max_predator_prob]), choice(
#             [i for i in self.graph.keys() if self.prey_q[i] == max_prey_prob])
#
#     def calculate_transition_probability_matrix(self):
#
#         shortest_distances = mp.get_shortest_distances_to_goals(self.graph, self.position, list(self.graph.keys())[:])
#         P = [[0 for i in range(self.config["GRAPH_SIZE"])] for i in range(self.config["GRAPH_SIZE"])]
#         for i in range(len(self.graph.keys())):
#             for j in self.graph[i]:  # for every neighbor j of i, the probability of moving from j to i
#                 P[i][j] = 0.4 * (1 / len(self.graph[j]))
#                 distances_from_j_neighbors = {x: shortest_distances[x] for x in self.graph[j]}
#                 min_dist = min(distances_from_j_neighbors.values())
#                 shortest_distances_from_j_neighbors = {x: y for (x, y) in distances_from_j_neighbors.items() if
#                                                        y <= min_dist}
#                 if i in shortest_distances_from_j_neighbors.keys():
#                     P[i][j] = P[i][j] + 0.6 * (1 / len(shortest_distances_from_j_neighbors.keys()))
#         return P
# AGENT 8 OLD CODE #####################################################################################################
# AGENT 7B OLD CODE ####################################################################################################
#
#     def belief_system(self, predator, prey):
#
#         if not self.found_predator:  # agent does know where the predator starts
#             # print("Running initialization...")
#             self.predator_q = [0 for i in range(self.config["GRAPH_SIZE"])]
#             self.predator_q[predator.position] = 1
#             self.found_predator = True
#         else:
#
#             predator_P = self.calculate_transition_probability_matrix()
#             self.predator_q = list(np.dot(predator_P, self.predator_q))
#
#             old_agent_pos_prob = self.predator_q[self.position]
#             self.predator_q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.predator_q))
#             self.predator_q[self.position] = 0
#
#         if not self.found_prey:
#             # initialization
#             self.prey_q = [1 / (self.config["GRAPH_SIZE"] - 1) for i in range(self.config["GRAPH_SIZE"])]
#             self.prey_q[self.position] = 0
#         else:
#             # calculate the probabilities of the nodes at each path depending on when the prey was last seen
#             self.prey_q = list(np.dot(self.prey_P, self.prey_q))
#
#             old_agent_pos_prob = self.prey_q[self.position]
#             self.prey_q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.prey_q))
#             self.prey_q[self.position] = 0
#
#         self.prey_q = au.normalize_probs(self.prey_q)
#         au.check_prob_sum(sum(self.prey_q))
#         self.predator_q = au.normalize_probs(self.predator_q)
#         au.check_prob_sum(sum(self.predator_q))
#
#         max_predator_prob = max(self.predator_q)
#         survey_spot = 0
#         if max_predator_prob != 1:
#             survey_spot = choice([i for i in self.graph.keys() if self.predator_q[i] == max_predator_prob])
#         else:
#             max_prey_prob = max(self.prey_q)
#             survey_spot = choice([i for i in self.graph.keys() if self.prey_q[i] == max_prey_prob])
#
#         defective = random.randrange(100) < 10
#
#         if defective:  # false negative
#             old_survey_spot_prob = self.prey_q[survey_spot]
#             self.prey_q[survey_spot] = 0
#             self.prey_q = list(map(lambda x: x / (1 - old_survey_spot_prob), self.prey_q))
#             old_survey_spot_prob = self.predator_q[survey_spot]
#             if old_survey_spot_prob != 1:
#                 self.predator_q[survey_spot] = 0
#                 self.predator_q = list(map(lambda x: x / (1 - old_survey_spot_prob), self.predator_q))
#
#         else:  # otherwise act normally
#             if survey_spot == prey.position:
#                 self.prey_q = [0 for i in range(self.config["GRAPH_SIZE"])]
#                 self.prey_q[survey_spot] = 1
#                 self.found_prey = True
#             else:
#                 old_survey_spot_prob = self.prey_q[survey_spot]
#                 self.prey_q[survey_spot] = 0
#                 self.prey_q = list(map(lambda x: x / (1 - old_survey_spot_prob), self.prey_q))
#
#             if survey_spot == predator.position:
#                 self.predator_q = [0 for i in range(self.config["GRAPH_SIZE"])]
#                 self.predator_q[survey_spot] = 1
#                 self.found_predator = True
#             else:
#                 old_survey_spot_prob = self.predator_q[survey_spot]
#                 self.predator_q[survey_spot] = 0
#                 self.predator_q = list(map(lambda x: x / (1 - old_survey_spot_prob), self.predator_q))
#
#         self.prey_q = au.normalize_probs(self.prey_q)
#         au.check_prob_sum(sum(self.prey_q))
#         self.predator_q = au.normalize_probs(self.predator_q)
#         au.check_prob_sum(sum(self.predator_q))
#
#         max_prey_prob = max(self.prey_q)
#         max_predator_prob = max(self.predator_q)
#         return choice([i for i in self.graph.keys() if self.predator_q[i] == max_predator_prob]), choice(
#             [i for i in self.graph.keys() if self.prey_q[i] == max_prey_prob])
#
#     def calculate_transition_probability_matrix(self):
#
#         shortest_distances = mp.get_shortest_distances_to_goals(self.graph, self.position, list(self.graph.keys())[:])
#         P = [[0 for i in range(self.config["GRAPH_SIZE"])] for i in range(self.config["GRAPH_SIZE"])]
#         for i in range(len(self.graph.keys())):
#             for j in self.graph[i]:  # for every neighbor j of i, the probability of moving from j to i
#                 P[i][j] = 0.4 * (1 / len(self.graph[j]))
#                 distances_from_j_neighbors = {x: shortest_distances[x] for x in self.graph[j]}
#                 min_dist = min(distances_from_j_neighbors.values())
#                 shortest_distances_from_j_neighbors = {x: y for (x, y) in distances_from_j_neighbors.items() if
#                                                        y <= min_dist}
#                 if i in shortest_distances_from_j_neighbors.keys():
#                     P[i][j] = P[i][j] + 0.6 * (1 / len(shortest_distances_from_j_neighbors.keys()))
#         return P
# AGENT 7B OLD CODE ####################################################################################################
# AGENT 8B OLD CODE ####################################################################################################
#     def belief_system(self, predator, prey):
#
#         if not self.found_predator:  # agent does know where the predator starts
#             # print("Running initialization...")
#             self.predator_q = [0 for i in range(self.config["GRAPH_SIZE"])]
#             self.predator_q[predator.position] = 1
#             self.found_predator = True
#         else:
#
#             predator_P = self.calculate_transition_probability_matrix()
#             self.predator_q = list(np.dot(predator_P, self.predator_q))
#
#             old_agent_pos_prob = self.predator_q[self.position]
#             self.predator_q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.predator_q))
#             self.predator_q[self.position] = 0
#
#         if not self.found_prey:
#             # initialization
#             self.prey_q = [1 / (self.config["GRAPH_SIZE"] - 1) for i in range(self.config["GRAPH_SIZE"])]
#             self.prey_q[self.position] = 0
#         else:
#             # calculate the probabilities of the nodes at each path depending on when the prey was last seen
#             self.prey_q = list(np.dot(self.prey_P, self.prey_q))
#
#             old_agent_pos_prob = self.prey_q[self.position]
#             self.prey_q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.prey_q))
#             self.prey_q[self.position] = 0
#
#         self.prey_q = au.normalize_probs(self.prey_q)
#         au.check_prob_sum(sum(self.prey_q))
#         self.predator_q = au.normalize_probs(self.predator_q)
#         au.check_prob_sum(sum(self.predator_q))
#
#         max_predator_prob = max(self.predator_q)
#         survey_spot = 0
#
#         defective = random.randrange(100) < 10
#
#         if max_predator_prob != 1:
#             survey_spot = choice([i for i in self.graph.keys() if self.predator_q[i] == max_predator_prob])
#         else:
#             max_prey_prob = max(self.prey_q)
#             survey_spot = choice([i for i in self.graph.keys() if self.prey_q[i] == max_prey_prob])
#
#         if defective:  # false negative
#             old_survey_spot_prob = self.prey_q[survey_spot]
#             self.prey_q[survey_spot] = 0
#             self.prey_q = list(map(lambda x: x / (1 - old_survey_spot_prob), self.prey_q))
#             old_survey_spot_prob = self.predator_q[survey_spot]
#             if old_survey_spot_prob != 1:
#                 self.predator_q[survey_spot] = 0
#                 self.predator_q = list(map(lambda x: x / (1 - old_survey_spot_prob), self.predator_q))
#         else:  # otherwise act normally
#             if survey_spot == prey.position:
#                 self.prey_q = [0 for i in range(self.config["GRAPH_SIZE"])]
#                 self.prey_q[survey_spot] = 1
#                 self.found_prey = True
#             else:
#                 old_survey_spot_prob = self.prey_q[survey_spot]
#                 self.prey_q[survey_spot] = 0
#                 self.prey_q = list(map(lambda x: x / (1 - old_survey_spot_prob), self.prey_q))
#
#             if survey_spot == predator.position:
#                 self.predator_q = [0 for i in range(self.config["GRAPH_SIZE"])]
#                 self.predator_q[survey_spot] = 1
#                 self.found_predator = True
#             else:
#                 old_survey_spot_prob = self.predator_q[survey_spot]
#                 self.predator_q[survey_spot] = 0
#                 self.predator_q = list(map(lambda x: x / (1 - old_survey_spot_prob), self.predator_q))
#
#         self.prey_q = au.normalize_probs(self.prey_q)
#         au.check_prob_sum(sum(self.prey_q))
#         self.predator_q = au.normalize_probs(self.predator_q)
#         au.check_prob_sum(sum(self.predator_q))
#
#         max_prey_prob = max(self.prey_q)
#         max_predator_prob = max(self.predator_q)
#         return choice([i for i in self.graph.keys() if self.predator_q[i] == max_predator_prob]), choice(
#             [i for i in self.graph.keys() if self.prey_q[i] == max_prey_prob])
#
#     def calculate_transition_probability_matrix(self):
#
#         shortest_distances = mp.get_shortest_distances_to_goals(self.graph, self.position, list(self.graph.keys())[:])
#         P = [[0 for i in range(self.config["GRAPH_SIZE"])] for i in range(self.config["GRAPH_SIZE"])]
#         for i in range(len(self.graph.keys())):
#             for j in self.graph[i]:  # for every neighbor j of i, the probability of moving from j to i
#                 P[i][j] = 0.4 * (1 / len(self.graph[j]))
#                 distances_from_j_neighbors = {x: shortest_distances[x] for x in self.graph[j]}
#                 min_dist = min(distances_from_j_neighbors.values())
#                 shortest_distances_from_j_neighbors = {x: y for (x, y) in distances_from_j_neighbors.items() if
#                                                        y <= min_dist}
#                 if i in shortest_distances_from_j_neighbors.keys():
#                     P[i][j] = P[i][j] + 0.6 * (1 / len(shortest_distances_from_j_neighbors.keys()))
#         return P
# AGENT 8B OLD CODE ####################################################################################################
# AGENT 7C OLD CODE ####################################################################################################
# 	def belief_system(self, predator, prey):
#
# 		if not self.found_predator: # agent does know where the predator starts
# 			# print("Running initialization...")
# 			self.predator_q = [0 for i in range(self.config["GRAPH_SIZE"])]
# 			self.predator_q[predator.position] = 1
# 			self.found_predator = True
# 		else:
#
# 			predator_P = self.calculate_transition_probability_matrix()
# 			self.predator_q = list(np.dot(predator_P, self.predator_q))
#
# 			old_agent_pos_prob = self.predator_q[self.position]
# 			self.predator_q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.predator_q))
# 			self.predator_q[self.position] = 0
#
# 		if not self.found_prey:
# 			# initialization
# 			self.prey_q = [1/(self.config["GRAPH_SIZE"] - 1) for i in range(self.config["GRAPH_SIZE"])]
# 			self.prey_q[self.position] = 0
# 		else:
# 			# calculate the probabilities of the nodes at each path depending on when the prey was last seen
# 			self.prey_q = list(np.dot(self.prey_P, self.prey_q))
#
# 			old_agent_pos_prob = self.prey_q[self.position]
# 			self.prey_q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.prey_q))
# 			self.prey_q[self.position] = 0
#
# 		self.prey_q = au.normalize_probs(self.prey_q)
# 		au.check_prob_sum(sum(self.prey_q))
# 		self.predator_q = au.normalize_probs(self.predator_q)
# 		au.check_prob_sum(sum(self.predator_q))
#
# 		max_predator_prob = max(self.predator_q)
# 		survey_spot = 0
# 		defective = random.randrange(100) < 10
# 		if max_predator_prob != 1:
# 			survey_spot = choice([i for i in self.graph.keys() if self.predator_q[i] == max_predator_prob])
# 		else:
# 			max_prey_prob = max(self.prey_q)
# 			survey_spot = choice([i for i in self.graph.keys() if self.prey_q[i] == max_prey_prob])
#
# 		if defective:
# 			old_survey_spot_prob = self.prey_q[survey_spot]
# 			self.prey_q[survey_spot] = 0.1*old_survey_spot_prob
# 			self.prey_q = list(map(lambda x: x / (0.1*old_survey_spot_prob + (1 - old_survey_spot_prob)), self.prey_q))
# 			old_survey_spot_prob = self.predator_q[survey_spot]
# 			if old_survey_spot_prob != 1:
# 				self.predator_q[survey_spot] = 0.1*old_survey_spot_prob
# 				self.predator_q = list(map(lambda x: x / (0.1*old_survey_spot_prob + (1 - old_survey_spot_prob)), self.predator_q))
#
# 		else:
# 			if survey_spot == prey.position:
# 				# P(prey in X | survey drone scans prey at S) = P(survey drone scans prey at S | prey in X)P(prey in X) / P(survey drone scans prey at S)
# 				# P(survey drone scans prey at S) = P(survey drone scans prey at S | prey in S)P(prey in S) + P(survey drone scans prey at S | prey not in S)P(prey not in S)
# 				# P(survey drone scans prey at S | prey in X) = 0 if X != S, 0.9 if X == S
# 				# if X != S
# 				# P(prey in X | survey drone scans prey at S) = 0*P(prey in X) / (0.9*P(prey in S) + 0*P(prey not in S)) = 0
# 				# if X == S
# 				# P(prey in X | survey drone scans prey at S) =  0.9*P(prey in S) / (0.9*P(prey in S) + 0*P(prey not in S)) = 1
#
# 				self.prey_q = [0 for i in range(self.config["GRAPH_SIZE"])]
# 				self.prey_q[survey_spot] = 1
# 				self.found_prey = True
# 			else:
# 				# P(prey in X | survey drone doesn't scan prey at S) = P(survey drone doesn't scan prey at S | prey in X)P(prey in X) / P(survey drone doesn't scan prey at S)
# 				# P(survey drone doesn't scan prey at S | prey in X) = 1 if X != S, 0.1 if X == S
# 				# P(survey drone doesn't scan prey at S) = P(survey drone doesn't scan prey at S | prey in S)P(prey in S) + P(survey drone doesn't scan prey at S | prey not in S)P(prey not in S)
# 				# = 0.1*P(prey in S) + 1*P(prey not in S)
# 				# for X != S
# 				# P(prey in X | survey drone doesn't scan prey at S) = 1*P(prey in X) / (0.1*P(prey in S) + 1*P(prey not in S))
# 				# for X == S
# 				# P(prey in X | survey drone doesn't scan prey at S) = 0.1*P(prey in S) / (0.1*P(prey in S) + 1*P(prey not in S))
#
# 				old_survey_spot_prob = self.prey_q[survey_spot]
# 				self.prey_q[survey_spot] = 0.1*old_survey_spot_prob
# 				self.prey_q = list(map(lambda x: x / (0.1*old_survey_spot_prob + (1 - old_survey_spot_prob)), self.prey_q))
#
# 				# print("Survey Spot: " + str(survey_spot))
#
# 			if survey_spot == predator.position:
# 				# P(pred in X | survey drone scans pred at S) = P(survey drone scans pred at S | pred in X)P(pred in X) / P(survey drone scans pred at S)
# 				# P(survey drone scans pred at S) = P(survey drone scans pred at S | pred in S)P(pred in S) + P(survey drone scans pred at S | pred not in S)P(pred not in S)
# 				# P(survey drone scans pred at S | pred in X) = 0 if X != S, 0.9 if X == S
# 				# if X != S
# 				# P(pred in X | survey drone scans pred at S) = 0*P(pred in X) / (0.9*P(pred in S) + 0*P(pred not in S)) = 0
# 				# if X == S
# 				# P(pred in X | survey drone scans pred at S) =  0.9*P(pred in S) / (0.9*P(pred in S) + 0*P(pred not in S)) = 1
# 				old_survey_spot_prob = self.predator_q[survey_spot]
# 				self.predator_q = [0 for i in range(self.config["GRAPH_SIZE"])]
# 				self.predator_q[survey_spot] = 1
# 				self.found_predator = True
# 				# print("Predator Found!")
# 			else:
# 				# P(pred in X | survey drone doesn't scan pred at S) = P(survey drone doesn't scan pred at S | pred in X)P(pred in X) / P(survey drone doesn't scan pred at S)
# 				# P(survey drone doesn't scan pred at S | pred in X) = 1 if X != S, 0.1 if X == S
# 				# P(survey drone doesn't scan pred at S) = P(survey drone doesn't scan pred at S | pred in S)P(pred in S) + P(survey drone doesn't scan pred at S | pred not in S)P(pred not in S)
# 				# = 0.1*P(pred in S) + 1*P(pred not in S)
# 				# for X != S
# 				# P(pred in X | survey drone doesn't scan pred at S) = 1*P(pred in X) / (0.1*P(pred in S) + 1*P(pred not in S))
# 				# for X == S
# 				# P(pred in X | survey drone doesn't scan pred at S) = 0.1*P(pred in S) / (0.1*P(pred in S) + 1*P(pred not in S))
# 				old_survey_spot_prob = self.predator_q[survey_spot]
# 				self.predator_q[survey_spot] = 0.1*old_survey_spot_prob
# 				self.predator_q = list(map(lambda x: x / (0.1*old_survey_spot_prob + (1 - old_survey_spot_prob)), self.predator_q))
#
# 		self.prey_q = au.normalize_probs(self.prey_q)
# 		au.check_prob_sum(sum(self.prey_q))
# 		self.predator_q = au.normalize_probs(self.predator_q)
# 		au.check_prob_sum(sum(self.predator_q))
#
# 		max_prey_prob = max(self.prey_q)
# 		max_predator_prob = max(self.predator_q)
# 		return choice([i for i in self.graph.keys() if self.predator_q[i] == max_predator_prob]), choice([i for i in self.graph.keys() if self.prey_q[i] == max_prey_prob])
#
# 	def calculate_transition_probability_matrix(self):
#
# 		shortest_distances = mp.get_shortest_distances_to_goals(self.graph, self.position, list(self.graph.keys())[:])
# 		P = [ [0 for i in range(self.config["GRAPH_SIZE"])] for i in range(self.config["GRAPH_SIZE"])]
# 		for i in range(len(self.graph.keys())):
# 			for j in self.graph[i]: # for every neighbor j of i, the probability of moving from j to i
# 				P[i][j] = 0.4*(1 / len(self.graph[j]))
# 				distances_from_j_neighbors = {x:shortest_distances[x] for x in self.graph[j]}
# 				min_dist = min(distances_from_j_neighbors.values())
# 				shortest_distances_from_j_neighbors = {x:y for (x, y) in distances_from_j_neighbors.items() if y <= min_dist}
# 				if i in shortest_distances_from_j_neighbors.keys():
# 					P[i][j] = P[i][j] + 0.6* (1 / len(shortest_distances_from_j_neighbors.keys()))
# 		return P
# AGENT 7C OLD CODE ####################################################################################################
# AGENT 8C OLD CODE ####################################################################################################
#     def belief_system(self, predator, prey):
#
#         if not self.found_predator: # agent does know where the predator starts
#             # print("Running initialization...")
#             self.predator_q = [0 for i in range(self.config["GRAPH_SIZE"])]
#             self.predator_q[predator.position] = 1
#             self.found_predator = True
#         else:
#
#             predator_P = self.calculate_transition_probability_matrix()
#             self.predator_q = list(np.dot(predator_P, self.predator_q))
#
#             old_agent_pos_prob = self.predator_q[self.position]
#             self.predator_q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.predator_q))
#             self.predator_q[self.position] = 0
#
#         if not self.found_prey:
#             # initialization
#             self.prey_q = [1/(self.config["GRAPH_SIZE"] - 1) for i in range(self.config["GRAPH_SIZE"])]
#             self.prey_q[self.position] = 0
#         else:
#             # calculate the probabilities of the nodes at each path depending on when the prey was last seen
#             self.prey_q = list(np.dot(self.prey_P, self.prey_q))
#
#             old_agent_pos_prob = self.prey_q[self.position]
#             self.prey_q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.prey_q))
#             self.prey_q[self.position] = 0
#
#         self.prey_q = au.normalize_probs(self.prey_q)
#         au.check_prob_sum(sum(self.prey_q))
#         self.predator_q = au.normalize_probs(self.predator_q)
#         au.check_prob_sum(sum(self.predator_q))
#
#         max_predator_prob = max(self.predator_q)
#         survey_spot = 0
#         defective = random.randrange(100) < 10
#         if max_predator_prob != 1:
#             survey_spot = choice([i for i in self.graph.keys() if self.predator_q[i] == max_predator_prob])
#         else:
#             max_prey_prob = max(self.prey_q)
#             survey_spot = choice([i for i in self.graph.keys() if self.prey_q[i] == max_prey_prob])
#
#         if defective:
#             old_survey_spot_prob = self.prey_q[survey_spot]
#             self.prey_q[survey_spot] = 0.1*old_survey_spot_prob
#             self.prey_q = list(map(lambda x: x / (0.1*old_survey_spot_prob + (1 - old_survey_spot_prob)), self.prey_q))
#             old_survey_spot_prob = self.predator_q[survey_spot]
#             if old_survey_spot_prob != 1:
#                 self.predator_q[survey_spot] = 0.1*old_survey_spot_prob
#                 self.predator_q = list(map(lambda x: x / (0.1*old_survey_spot_prob + (1 - old_survey_spot_prob)), self.predator_q))
#
#         else:
#             if survey_spot == prey.position:
#                 # P(prey in X | survey drone scans prey at S) = P(survey drone scans prey at S | prey in X)P(prey in X) / P(survey drone scans prey at S)
#                 # P(survey drone scans prey at S) = P(survey drone scans prey at S | prey in S)P(prey in S) + P(survey drone scans prey at S | prey not in S)P(prey not in S)
#                 # P(survey drone scans prey at S | prey in X) = 0 if X != S, 0.9 if X == S
#                 # if X != S
#                 # P(prey in X | survey drone scans prey at S) = 0*P(prey in X) / (0.9*P(prey in S) + 0*P(prey not in S)) = 0
#                 # if X == S
#                 # P(prey in X | survey drone scans prey at S) =  0.9*P(prey in S) / (0.9*P(prey in S) + 0*P(prey not in S)) = 1
#
#                 self.prey_q = [0 for i in range(self.config["GRAPH_SIZE"])]
#                 self.prey_q[survey_spot] = 1
#                 self.found_prey = True
#             else:
#                 # P(prey in X | survey drone doesn't scan prey at S) = P(survey drone doesn't scan prey at S | prey in X)P(prey in X) / P(survey drone doesn't scan prey at S)
#                 # P(survey drone doesn't scan prey at S | prey in X) = 1 if X != S, 0.1 if X == S
#                 # P(survey drone doesn't scan prey at S) = P(survey drone doesn't scan prey at S | prey in S)P(prey in S) + P(survey drone doesn't scan prey at S | prey not in S)P(prey not in S)
#                 # = 0.1*P(prey in S) + 1*P(prey not in S)
#                 # for X != S
#                 # P(prey in X | survey drone doesn't scan prey at S) = 1*P(prey in X) / (0.1*P(prey in S) + 1*P(prey not in S))
#                 # for X == S
#                 # P(prey in X | survey drone doesn't scan prey at S) = 0.1*P(prey in S) / (0.1*P(prey in S) + 1*P(prey not in S))
#
#                 old_survey_spot_prob = self.prey_q[survey_spot]
#                 self.prey_q[survey_spot] = 0.1*old_survey_spot_prob
#                 self.prey_q = list(map(lambda x: x / (0.1*old_survey_spot_prob + (1 - old_survey_spot_prob)), self.prey_q))
#
#             # print("Survey Spot: " + str(survey_spot))
#
#             if survey_spot == predator.position:
#                 # P(pred in X | survey drone scans pred at S) = P(survey drone scans pred at S | pred in X)P(pred in X) / P(survey drone scans pred at S)
#                 # P(survey drone scans pred at S) = P(survey drone scans pred at S | pred in S)P(pred in S) + P(survey drone scans pred at S | pred not in S)P(pred not in S)
#                 # P(survey drone scans pred at S | pred in X) = 0 if X != S, 0.9 if X == S
#                 # if X != S
#                 # P(pred in X | survey drone scans pred at S) = 0*P(pred in X) / (0.9*P(pred in S) + 0*P(pred not in S)) = 0
#                 # if X == S
#                 # P(pred in X | survey drone scans pred at S) =  0.9*P(pred in S) / (0.9*P(pred in S) + 0*P(pred not in S)) = 1
#                 old_survey_spot_prob = self.predator_q[survey_spot]
#                 self.predator_q = [0 for i in range(self.config["GRAPH_SIZE"])]
#                 self.predator_q[survey_spot] = 1
#                 self.found_predator = True
#             # print("Predator Found!")
#             else:
#                 # P(pred in X | survey drone doesn't scan pred at S) = P(survey drone doesn't scan pred at S | pred in X)P(pred in X) / P(survey drone doesn't scan pred at S)
#                 # P(survey drone doesn't scan pred at S | pred in X) = 1 if X != S, 0.1 if X == S
#                 # P(survey drone doesn't scan pred at S) = P(survey drone doesn't scan pred at S | pred in S)P(pred in S) + P(survey drone doesn't scan pred at S | pred not in S)P(pred not in S)
#                 # = 0.1*P(pred in S) + 1*P(pred not in S)
#                 # for X != S
#                 # P(pred in X | survey drone doesn't scan pred at S) = 1*P(pred in X) / (0.1*P(pred in S) + 1*P(pred not in S))
#                 # for X == S
#                 # P(pred in X | survey drone doesn't scan pred at S) = 0.1*P(pred in S) / (0.1*P(pred in S) + 1*P(pred not in S))
#                 old_survey_spot_prob = self.predator_q[survey_spot]
#                 self.predator_q[survey_spot] = 0.1*old_survey_spot_prob
#                 self.predator_q = list(map(lambda x: x / (0.1*old_survey_spot_prob + (1 - old_survey_spot_prob)), self.predator_q))
#
#         self.prey_q = au.normalize_probs(self.prey_q)
#         au.check_prob_sum(sum(self.prey_q))
#         self.predator_q = au.normalize_probs(self.predator_q)
#         au.check_prob_sum(sum(self.predator_q))
#
#         max_prey_prob = max(self.prey_q)
#         max_predator_prob = max(self.predator_q)
#         return choice([i for i in self.graph.keys() if self.predator_q[i] == max_predator_prob]), choice([i for i in self.graph.keys() if self.prey_q[i] == max_prey_prob])
#
#     def calculate_transition_probability_matrix(self):
#
#         shortest_distances = mp.get_shortest_distances_to_goals(self.graph, self.position, list(self.graph.keys())[:])
#         P = [ [0 for i in range(self.config["GRAPH_SIZE"])] for i in range(self.config["GRAPH_SIZE"])]
#         for i in range(len(self.graph.keys())):
#             for j in self.graph[i]: # for every neighbor j of i, the probability of moving from j to i
#                 P[i][j] = 0.4*(1 / len(self.graph[j]))
#                 distances_from_j_neighbors = {x:shortest_distances[x] for x in self.graph[j]}
#                 min_dist = min(distances_from_j_neighbors.values())
#                 shortest_distances_from_j_neighbors = {x:y for (x, y) in distances_from_j_neighbors.items() if y <= min_dist}
#                 if i in shortest_distances_from_j_neighbors.keys():
#                     P[i][j] = P[i][j] + 0.6* (1 / len(shortest_distances_from_j_neighbors.keys()))
#         return P
# AGENT 8C OLD CODE ####################################################################################################