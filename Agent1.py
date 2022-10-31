import MapUtils as mp
from random import sample 

class Agent1:
	def __init__(self, graph, start, config):
		self.position = start
		self.graph = graph
		self.config = config

	def update(self, predator, prey):
		# print("Agent Running...")
		# print("Analyzing " + str(predator.position) + " and " + str(prey.position))
		neighbors_and_self = self.graph[self.position][:]
		neighbors_and_self.append(self.position)

		predator_distances = mp.getShortestDistancesToGoals(self.graph, predator.position, neighbors_and_self[:])
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

		# print("Closer to prey" + str(closer_to_prey))
		# print("Same to prey" + str(same_to_prey))
		# print("Further to prey" + str(far_from_prey))
		# print("Closer to pred" + str(closer_to_pred))
		# print("Same to pred" + str(same_to_prey))
		# print("Further to pred" + str(far_from_pred))
		# print("Closer to Prey and Further From Predator: " + str(closer_to_prey_and_further_from_pred))
		# print("Closer to Prey and Same From Predator: " + str(closer_to_prey_and_same_from_pred))
		# print("Same to Prey and Further From Predator: " + str(same_to_prey_and_further_from_pred))
		# print("Same to Prey and Same From Predator: " + str(same_to_prey_and_same_from_pred))


		if len(closer_to_prey_and_further_from_pred) != 0:
			self.position = sample(closer_to_prey_and_further_from_pred, 1)[0]
		elif len(closer_to_prey_and_same_from_pred) != 0:
			self.position = sample(closer_to_prey_and_same_from_pred, 1)[0]
		elif len(same_to_prey_and_further_from_pred) != 0:
			self.position = sample(same_to_prey_and_further_from_pred, 1)[0]
		elif len(same_to_prey_and_same_from_pred) != 0:
			self.position = sample(same_to_prey_and_same_from_pred, 1)[0]
		elif len(far_from_pred) != 0:
			self.position = sample(far_from_pred, 1)[0]
		elif len(same_to_pred) != 0:
			self.position = sample(same_to_pred, 1)[0]

		# flag = False
		# if smallest_prey <= cur_dist_prey:
		# 	if predator_distances[smallest_prey_pos] >= cur_dist_pred:
		# 		self.position = smallest_prey_pos
		# 		flag = True
		# 		# print("Reached")
		# if not flag and largest_pred >= cur_dist_pred:
		# 	self.position = largest_pred_pos
    	
		return 1 if self.position == prey.position else -1 if self.position == predator.position else 0


		# closer_to_prey = set([x for x in prey_distances.keys() if prey_distances[x] < cur_dist_prey])
		# same_to_prey = set([x for x in prey_distances.keys() if prey_distances[x] == cur_dist_prey])
		# far_from_prey = set([x for x in prey_distances.keys() if prey_distances[x] > cur_dist_prey])
		# closer_to_pred = set([x for x in predator_distances.keys() if predator_distances[x] < cur_dist_prey])
		# same_to_pred = set([x for x in predator_distances.keys() if predator_distances[x] == cur_dist_prey])
		# far_from_pred = set([x for x in predator_distances.keys() if predator_distances[x] > cur_dist_prey])

		# closer_to_prey_and_further_from_pred = closer_to_prey.intersection(far_from_prey)
		# closer_to_prey_and_same_from_pred = closer_to_prey.intersection(same_to_pred)
		# same_to_prey_and_further_from_pred = same_to_prey.intersection(far_from_prey)
		# same_to_prey_and_same_from_pred = same_to_prey.intersection(same_to_pred)



		# if len(closer_to_prey_and_further_from_pred) != 0:
		# 	self.position = sample(closer_to_prey_and_further_from_pred, 1)[0]
		# 	print("Moving closer to prey and further from pred")
		# elif len(closer_to_prey_and_same_from_pred) != 0:
		# 	self.position = sample(closer_to_prey_and_same_from_pred, 1)[0]
		# 	print("Moving closer to prey and same from pred")
		# elif len(same_to_prey_and_further_from_pred) != 0:
		# 	self.position = sample(same_to_prey_and_further_from_pred, 1)[0]
		# 	print("Moving same to prey and further from pred")
		# elif len(same_to_prey_and_same_from_pred) != 0:
		# 	self.position = sample(same_to_prey_and_same_from_pred, 1)[0]
		# 	print("Moving same to prey and same from pred")
		# elif len(far_from_pred) != 0:
		# 	self.position = sample(far_from_pred, 1)[0]
		# 	print("Moving farther to prey and further from pred")
		# elif len(same_to_pred) != 0:
		# 	self.position = sample(same_to_pred, 1)[0]
		# 	print("Moving farther to prey and same from pred")

