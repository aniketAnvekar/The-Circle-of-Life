import MapUtils as mp
from random import choice
import numpy as np
import AgentUtils as au


class Agent7:
	def __init__(self, graph, start, config):
		self.position = start
		self.graph = graph
		self.config = config
		self.prey_q = [1 / (self.config["GRAPH_SIZE"]) for i in
					   range(self.config["GRAPH_SIZE"])]  # vector containing probabilities of where the prey is
		self.predator_q = []
		self.found_prey = False
		self.prey_P = [[0 for i in range(self.config["GRAPH_SIZE"])] for i in
					   range(self.config["GRAPH_SIZE"])]  # transition probabilities from j to i
		for i in graph.keys():
			self.prey_P[i][i] = 1 / (len(self.graph[i]) + 1)
			for j in graph[i]:
				self.prey_P[i][j] = 1 / (len(self.graph[j]) + 1)
		self.found_predator = False
		self.predator_q = [1 / (self.config["GRAPH_SIZE"]) for i in
						   range(self.config["GRAPH_SIZE"])]  # vector containing probabilities of where the prey is

	def checkProbSum(self, su):
		if abs(1 - su) < 0.000000001:  # 0.000000000000001
			return
		print("BELIEF SYSTEM FAULTY")
		exit()

	def belief_system(self, predator, prey):

		if not self.found_predator:  # agent does know where the predator starts
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
			self.prey_q = [1 / (self.config["GRAPH_SIZE"] - 1) for i in range(self.config["GRAPH_SIZE"])]
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
		if max_predator_prob != 1:
			survey_spot = choice([i for i in self.graph.keys() if self.predator_q[i] == max_predator_prob])
		else:
			max_prey_prob = max(self.prey_q)
			survey_spot = choice([i for i in self.graph.keys() if self.prey_q[i] == max_prey_prob])

		if survey_spot == prey.position:
			self.prey_q = [0 for i in range(self.config["GRAPH_SIZE"])]
			self.prey_q[survey_spot] = 1
			self.found_prey = True
		else:
			old_survey_spot_prob = self.prey_q[survey_spot]
			self.prey_q[survey_spot] = 0
			self.prey_q = list(map(lambda x: x / (1 - old_survey_spot_prob), self.prey_q))

		if survey_spot == predator.position:
			self.predator_q = [0 for i in range(self.config["GRAPH_SIZE"])]
			self.predator_q[survey_spot] = 1
			self.found_predator = True
		else:
			old_survey_spot_prob = self.predator_q[survey_spot]
			self.predator_q[survey_spot] = 0
			self.predator_q = list(map(lambda x: x / (1 - old_survey_spot_prob), self.predator_q))

		self.prey_q = au.normalize_probs(self.prey_q)
		au.check_prob_sum(sum(self.prey_q))
		self.predator_q = au.normalize_probs(self.predator_q)
		au.check_prob_sum(sum(self.predator_q))

		max_prey_prob = max(self.prey_q)
		max_predator_prob = max(self.predator_q)
		return choice([i for i in self.graph.keys() if self.predator_q[i] == max_predator_prob]), choice(
			[i for i in self.graph.keys() if self.prey_q[i] == max_prey_prob])

	def calculate_transition_probability_matrix(self):

		shortest_distances = mp.get_shortest_distances_to_goals(self.graph, self.position, list(self.graph.keys())[:])
		P = [[0 for i in range(self.config["GRAPH_SIZE"])] for i in range(self.config["GRAPH_SIZE"])]
		for i in range(len(self.graph.keys())):
			for j in self.graph[i]:  # for every neighbor j of i, the probability of moving from j to i
				P[i][j] = 0.4 * (1 / len(self.graph[j]))
				distances_from_j_neighbors = {x: shortest_distances[x] for x in self.graph[j]}
				min_dist = min(distances_from_j_neighbors.values())
				shortest_distances_from_j_neighbors = {x: y for (x, y) in distances_from_j_neighbors.items() if
													   y <= min_dist}
				if i in shortest_distances_from_j_neighbors.keys():
					P[i][j] = P[i][j] + 0.6 * (1 / len(shortest_distances_from_j_neighbors.keys()))
		return P

	def update(self, predator, prey):

		estimated_predator_position, estimated_prey_position = self.belief_system(predator, prey)

		return au.basic_update_agent(self, predator, prey, estimated_predator_position=estimated_predator_position,
									 estimated_prey_position=estimated_prey_position)
