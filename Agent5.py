import MapUtils as mp
from random import choice 
import numpy as np

class Agent5:
	def __init__(self, graph, start, config):
		self.position = start
		self.graph = graph
		self.config = config
		self.found_predator = False
		self.q = [1/(self.config["GRAPH_SIZE"]) for i in range(self.config["GRAPH_SIZE"])] # vector containing probabilities of where the prey is

	def checkProbSum(self, su):
		if abs(1 - su) < 0.000000000001: # 0.000000000000001
			return
		print("BELIEF SYSTEM FAULTY")
		exit()

	def belief_system(self, predator):

		# print("Total Belief Sum: " + str(sum(self.q)))

		self.checkProbSum(sum(self.q))

		if not self.found_predator: # agent does not know where the predator starts 
			# initialization
			self.q = [1/(self.config["GRAPH_SIZE"] - 1) for i in range(self.config["GRAPH_SIZE"])]
			self.q[self.position] = 0

		# if not self.found_predator: # agent does know where the predator starts
		# 	print("Running initialization...")
		# 	self.q = [0 for i in range(self.config["GRAPH_SIZE"])]
		# 	self.q[predator.position] = 1
		else:
			# update for predator
			# print("Predator gunning for " + str(self.position))
			# # print(self.graph)
			P = self.calculate_transition_probability_matrix()
			self.q = list(np.dot(P, self.q))

			# print("Total Belief Sum Multiply Matrix: " + str(sum(self.q)))
			# print(self.q)
			self.checkProbSum(sum(self.q))

			old_agent_pos_prob = self.q[self.position]
			self.q = list(map(lambda x: x / (1 - old_agent_pos_prob), self.q))
			self.q[self.position] = 0

		# print("Total Belief Sum Account for Current Agent Pos: " + str(sum(self.q)))

		# print("Total Belief Sum: " + str(sum(self.q)))

		self.checkProbSum(sum(self.q))

		max_prob = max(self.q)
		survey_spot = choice([i for i in self.graph.keys() if self.q[i] == max_prob])

		# print("Survey Spot: " + str(survey_spot))

		if survey_spot == predator.position:
			self.q = [0 for i in range(self.config["GRAPH_SIZE"])]
			self.q[survey_spot] = 1
			self.found_predator = True
			# print("Predator Found!")
		else:
			old_survey_spot_prob = self.q[survey_spot]
			# print(survey_spot)
			# print(predator.position)
			# print(self.graph[survey_spot])
			# print(self.graph[predator.position])
			self.q[survey_spot] = 0
			self.q = list(map(lambda x: x / (1 - old_survey_spot_prob), self.q))
		# 	print("Predator Not Found!")
		# 	print(self.q)

		# print("Total Belief Sum: " + str(sum(self.q)))
		self.checkProbSum(sum(self.q))

		max_prob = max(self.q)
		return choice([i for i in self.graph.keys() if self.q[i] == max_prob]) 


	def calculate_transition_probability_matrix(self):
		possible_locations = list(filter(lambda x: self.q[x] != 0, self.graph.keys()))
		# print(possible_locations)
		P = [ [0 for i in range(self.config["GRAPH_SIZE"])] for i in range(self.config["GRAPH_SIZE"])]
		for possible_location in possible_locations:
			# print(possible_location)
			# calculate shortest distances to goal
			# shortest_distances = {}
			# mp.shortest_distances_to_goal(self.graph, set(), possible_location, self.position, shortest_distances)
			# print(shortest_distances)
			# keep only neighbors
			# shortest_distances = {x:y for (x,y) in shortest_distances.items() if x in self.graph[possible_location]}
			# print(shortest_distances)
			# find minimum and filter for only remaining
			# min_dist = min(shortest_distances.values())
			# shortest_distances = {x:y for (x, y) in shortest_distances.items() if y <= min_dist}
			
			shortest_distances = mp.getShortestDistancesToGoals(self.graph, self.position, self.graph[possible_location][:])
			# print(shortest_distances)
			min_dist = min(shortest_distances.values())
			shortest_distances = {x:y for (x, y) in shortest_distances.items() if y <= min_dist}
			# print(shortest_distances)

			for neighbor in shortest_distances.keys():
				P[neighbor][possible_location] = 1 / (len(shortest_distances.keys())) #1 / possible locations
		return P


	def update(self, predator, prey):

		estimated_predator_position = self.belief_system(predator)
		# print("Estimated predator position: " + str(estimated_predator_position))

		neighbors_and_self = self.graph[self.position][:]
		neighbors_and_self.append(self.position)

		predator_distances = mp.getShortestDistancesToGoals(self.graph, estimated_predator_position, neighbors_and_self[:])
		prey_distances = mp.getShortestDistancesToGoals(self.graph, prey.position, neighbors_and_self[:])

		# print(predator_distances)
		# print(prey_distances)
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

		# print(smallest_prey, predator_distances[smallest_prey_pos], cur_dist_prey, cur_dist_pred)
		# print(prey_distances[largest_pred_pos], largest_pred, cur_dist_prey, cur_dist_pred)

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

