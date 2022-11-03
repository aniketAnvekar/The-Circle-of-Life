import MapUtils as mp
from random import choice 
import numpy as np 

class Agent7D:

	# This is Agent 7 with a defective survey drone 


	def __init__(self, graph, start, config):
		self.position = start
		self.graph = graph
		self.config = config
		self.prey_q = [1/(self.config["GRAPH_SIZE"]) for i in range(self.config["GRAPH_SIZE"])] # vector containing probabilities of where the prey is
		self.predator_q = []
		self.found_prey = False
		self.prey_P = [ [0 for i in range(self.config["GRAPH_SIZE"])] for i in range(self.config["GRAPH_SIZE"])] # transition probabilities from j to i
		for i in graph.keys():
			self.prey_P[i][i] = 1/(len(self.graph[i]) + 1)
			for j in graph[i]:
				self.prey_P[i][j] = 1/(len(self.graph[j]) + 1)
		self.found_predator = False
		self.predator_q = [1/(self.config["GRAPH_SIZE"]) for i in range(self.config["GRAPH_SIZE"])] # vector containing probabilities of where the prey is


	def checkProbSum(self, su):
		if abs(1 - su) < 0.000000000001: # 0.000000000000001
			return
		print("BELIEF SYSTEM FAULTY")
		exit()


	def belief_system(self, predator, prey):

		self.checkProbSum(sum(self.prey_q))
		self.checkProbSum(sum(self.predator_q))

		if not self.found_predator: # agent does not know where the predator starts 
			# initialization
			self.predator_q = [1/(self.config["GRAPH_SIZE"] - 1) for i in range(self.config["GRAPH_SIZE"])]
			self.predator_q[self.position] = 0
		# if not self.found_predator: # agent does know where the predator starts
		# 	# print("Running initialization...")
		# 	self.predator_q = [0 for i in range(self.config["GRAPH_SIZE"])]
		# 	self.predator_q[predator.position] = 1
		# 	self.found_predator = True
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

		self.checkProbSum(sum(self.prey_q))
		self.checkProbSum(sum(self.predator_q))

		max_predator_prob = max(self.predator_q)
		if max_predator_prob != 1:
			max_prey_prob = max(self.prey_q)
			prey_survey_spot = choice([i for i in self.graph.keys() if self.prey_q[i] == max_prey_prob])
			if prey_survey_spot == prey.position:
				# P(prey in X | survey drone scans prey at S) = P(survey drone scans prey at S | prey in X)P(prey in X) / P(survey drone scans prey at S)
				# P(survey drone scans prey at S) = P(survey drone scans prey at S | prey in S)P(prey in S) + P(survey drone scans prey at S | prey not in S)P(prey not in S)
				# P(survey drone scans prey at S | prey in X) = 0 if X != S, 0.9 if X == S
				# if X != S
				# P(prey in X | survey drone scans prey at S) = 0*P(prey in X) / (0.9*P(prey in S) + 0*P(prey not in S)) = 0
				# if X == S
				# P(prey in X | survey drone scans prey at S) =  0.9*P(prey in S) / (0.9*P(prey in S) + 0*P(prey not in S)) = 1

				self.prey_q = [0 for i in range(self.config["GRAPH_SIZE"])]
				self.prey_q[prey_survey_spot] = 1
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

				old_survey_spot_prob = self.prey_q[prey_survey_spot]
				self.prey_q[prey_survey_spot] = 0.1*old_survey_spot_prob
				self.prey_q = list(map(lambda x: x / (0.1*old_survey_spot_prob + (1 - old_survey_spot_prob)), self.prey_q))
		else:
			predator_survey_spot = choice([i for i in self.graph.keys() if self.predator_q[i] == max_predator_prob])
			# print("Survey Spot: " + str(survey_spot))

			if predator_survey_spot == predator.position:
				# P(pred in X | survey drone scans pred at S) = P(survey drone scans pred at S | pred in X)P(pred in X) / P(survey drone scans pred at S)
				# P(survey drone scans pred at S) = P(survey drone scans pred at S | pred in S)P(pred in S) + P(survey drone scans pred at S | pred not in S)P(pred not in S)
				# P(survey drone scans pred at S | pred in X) = 0 if X != S, 0.9 if X == S
				# if X != S
				# P(pred in X | survey drone scans pred at S) = 0*P(pred in X) / (0.9*P(pred in S) + 0*P(pred not in S)) = 0
				# if X == S
				# P(pred in X | survey drone scans pred at S) =  0.9*P(pred in S) / (0.9*P(pred in S) + 0*P(pred not in S)) = 1
				old_survey_spot_prob = self.predator_q[predator_survey_spot]
				self.predator_q = [0 for i in range(self.config["GRAPH_SIZE"])]
				self.predator_q[predator_survey_spot] = 1
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
				old_survey_spot_prob = self.predator_q[predator_survey_spot]
				self.predator_q[predator_survey_spot] = 0.1*old_survey_spot_prob
				self.predator_q = list(map(lambda x: x / (0.1*old_survey_spot_prob + (1 - old_survey_spot_prob)), self.predator_q))

		self.checkProbSum(sum(self.prey_q))
		self.checkProbSum(sum(self.predator_q))

		max_prey_prob = max(self.prey_q)
		max_predator_prob = max(self.predator_q)
		return choice([i for i in self.graph.keys() if self.predator_q[i] == max_predator_prob]), choice([i for i in self.graph.keys() if self.prey_q[i] == max_prey_prob]) 

	def calculate_transition_probability_matrix(self):
		possible_locations = list(filter(lambda x: self.predator_q[x] != 0, self.graph.keys()))
		# print(possible_locations)
		P = [ [0 for i in range(self.config["GRAPH_SIZE"])] for i in range(self.config["GRAPH_SIZE"])]
		for possible_location in possible_locations:
			
			shortest_distances = mp.getShortestDistancesToGoals(self.graph, self.position, self.graph[possible_location][:])
			# print(shortest_distances)
			min_dist = min(shortest_distances.values())
			shortest_distances = {x:y for (x, y) in shortest_distances.items() if y <= min_dist}
			# print(shortest_distances)

			for neighbor in shortest_distances.keys():
				P[neighbor][possible_location] = 1 / (len(shortest_distances.keys())) #1 / possible locations
		return P


	def update(self, predator, prey):

		estimated_predator_position, estimated_prey_position = self.belief_system(predator, prey)
		# print("Agent Running...")
		# print("Analyzing " + str(predator.position) + " and " + str(prey.position))
		neighbors_and_self = self.graph[self.position][:]
		neighbors_and_self.append(self.position)

		predator_distances = mp.getShortestDistancesToGoals(self.graph, estimated_predator_position, neighbors_and_self[:])
		prey_distances = mp.getShortestDistancesToGoals(self.graph, estimated_prey_position, neighbors_and_self[:])

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






		