import MapUtils as mp
from random import choice
import numpy as np

class Agent3:
	def __init__(self, graph, start, config):
		self.position = start
		self.graph = graph
		self.config = config
		self.found_prey = False
		self.q = [1/(self.config["GRAPH_SIZE"]) for i in range(self.config["GRAPH_SIZE"])] # vector containing probabilities of where the prey is
		self.P = [ [0 for i in range(self.config["GRAPH_SIZE"])] for i in range(self.config["GRAPH_SIZE"])] # transition probabilities from j to i
		for i in graph.keys():
			self.P[i][i] = 1/(len(self.graph[i]) + 1)
			for j in graph[i]:
				self.P[i][j] = 1/(len(self.graph[j]) + 1)

	def checkProbSum(self, su):
		if abs(1 - su) < 0.00000000000001: # 0.000000000000001
			return
		print("BELIEF SYSTEM FAULTY")
		exit()

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
			self.checkProbSum(sum(self.q))

			old_agent_pos_prob = self.q[self.position]
			self.q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.q))
			self.q[self.position] = 0
			# 	print("Total Belief Sum Account for Current Agent Pos: " + str(sum(self.q)))
			self.checkProbSum(sum(self.q))
		
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
		self.checkProbSum(sum(self.q))

		max_prob = max(self.q)
		return choice([i for i in self.graph.keys() if self.q[i] == max_prob])


	def update(self, predator, prey):
		estimated_prey_position = self.belief_system(prey)
		# print("Estimated position: " + str(estimated_prey_position))

		neighbors_and_self = self.graph[self.position][:]
		neighbors_and_self.append(self.position)

		predator_distances = mp.getShortestDistancesToGoals(self.graph, predator.position, neighbors_and_self[:])
		prey_distances = mp.getShortestDistancesToGoals(self.graph, estimated_prey_position, neighbors_and_self[:])

		cur_dist_pred = predator_distances[self.position]
		cur_dist_prey = prey_distances[self.position]
		smallest_prey = self.config["GRAPH_SIZE"] + 1
		smallest_prey_pos = -1
		largest_pred = -1
		largest_pred_pos = -1

		for position in predator_distances.keys():

			if prey_distances[position] <= smallest_prey:
				smallest_prey = prey_distances[position]
				smallest_prey_pos = position
			if predator_distances[position] >= largest_pred:
				largest_pred = predator_distances[position]
				largest_pred_pos = position

		closer_to_prey = set([x for x in prey_distances.keys() if prey_distances[x] < cur_dist_prey and x != self.position] )
		same_to_prey = set([x for x in prey_distances.keys() if prey_distances[x] == cur_dist_prey and x != self.position])
		far_from_prey = set([x for x in prey_distances.keys() if prey_distances[x] > cur_dist_prey and x != self.position])
		closer_to_pred = set([x for x in predator_distances.keys() if predator_distances[x] < cur_dist_pred and x != self.position])
		same_to_pred = set([x for x in predator_distances.keys() if predator_distances[x] == cur_dist_pred and x != self.position])
		far_from_pred = set([x for x in predator_distances.keys() if predator_distances[x] > cur_dist_pred and x != self.position])

		closer_to_prey_and_further_from_pred = closer_to_prey.intersection(far_from_pred)
		closer_to_prey_and_same_from_pred = closer_to_prey.intersection(same_to_pred)
		same_to_prey_and_further_from_pred = same_to_prey.intersection(far_from_pred)
		same_to_prey_and_same_from_pred = same_to_prey.intersection(same_to_pred)


		if len(closer_to_prey_and_further_from_pred) != 0:
			self.position = choice(list(closer_to_prey_and_further_from_pred))
		elif len(closer_to_prey_and_same_from_pred) != 0:
			self.position = choice(list(closer_to_prey_and_same_from_pred))
		elif len(same_to_prey_and_further_from_pred) != 0:
			self.position = choice(list(same_to_prey_and_further_from_pred))
		elif len(same_to_prey_and_same_from_pred) != 0:
			self.position = choice(list(same_to_prey_and_same_from_pred))
		elif len(far_from_pred) != 0:
			self.position = choice(list(far_from_pred))
		elif len(same_to_pred) != 0:
			self.position = choice(list(same_to_pred))
    	
		return 1 if self.position == prey.position else -1 if self.position == predator.position else 0

