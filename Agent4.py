import MapUtils as mp
from random import choice
import AgentUtils as au
import numpy as np


class Agent4:
	def __init__(self, graph, start, config):
		self.position = start
		self.graph = graph
		self.config = config
		self.visited = [0 for _ in self.graph.keys()]
		self.found_prey = False
		self.q = [1 / (self.config["GRAPH_SIZE"]) for _ in
				  range(self.config["GRAPH_SIZE"])]  # vector containing probabilities of where the prey is
		self.P = [[0 for _ in range(self.config["GRAPH_SIZE"])] for _ in
				  range(self.config["GRAPH_SIZE"])]  # transition probabilities from j to i
		for i in graph.keys():
			self.P[i][i] = 1 / (len(self.graph[i]) + 1)
			for j in graph[i]:
				self.P[i][j] = 1 / (len(self.graph[j]) + 1)

	def belief_system(self, prey):

		# print("Total Belief Sum: " + str(sum(self.q)))

		if not self.found_prey:
			# initialization
			self.q = [1/(self.config["GRAPH_SIZE"] - 1) for i in range(self.config["GRAPH_SIZE"])]
			self.q[self.position] = 0
		else:
			# calculate the probabilities of the nodes at each path depending on when the prey was last seen
			self.q = list(np.dot(self.P, self.q))
			# print("Total Belief Sum Multiply Matrix: " + str(sum(self.q)))
			# print(self.q)

			old_agent_pos_prob = self.q[self.position]
			self.q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.q))
			self.q[self.position] = 0
		# 	print("Total Belief Sum Account for Current Agent Pos: " + str(sum(self.q)))

		self.q = au.normalize_probs(self.q)
		au.check_prob_sum(sum(self.q))
		# print("Total Belief Sum: " + str(sum(self.q)))

		#survey a node
		max_prob = max(self.q)
		survey_spot = choice([i for i in self.graph.keys() if self.q[i] == max_prob])
		# print("Survey Spot: " + str(survey_spot))
		if survey_spot == prey.position:
			self.q = [0 for i in range(self.config["GRAPH_SIZE"])]
			self.q[survey_spot] = 1
			self.found_prey = True
		# print("Prey Found!")
		else:
			old_survey_spot_prob = self.q[survey_spot]
			self.q[survey_spot] = 0
			self.q = list(map(lambda x: x / (1 - old_survey_spot_prob), self.q))
		# print("Prey Not Found!")
		# print(self.q)

		# print("Total Belief Sum: " + str(sum(self.q)))
		self.q = au.normalize_probs(self.q)
		au.check_prob_sum(sum(self.q))

		max_prob = max(self.q)
		return choice([i for i in self.graph.keys() if self.q[i] == max_prob])

	def update(self, predator, prey):
		estimated_prey_position = self.belief_system(prey)
		options = list(filter(lambda x: self.q[x] == self.q[estimated_prey_position], self.graph.keys()))
		if len(options) != 1:
			distances = mp.get_shortest_distances_to_goals(self.graph, predator.position, options)
			longest = max(distances.values())
			estimated_prey_position = choice([i for i in distances.keys() if distances[i] == longest])

		return au.advanced_update_agent(self, predator, prey, estimated_prey_position=estimated_prey_position)
