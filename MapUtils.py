import queue as q
from collections import deque
from random import shuffle


def getShortestDistancesToGoals(graph, start, goals):
	queue = q.Queue()
	queue.put(start)
	visited = set()
	shortestDistances = {}
	lengths = {}

	lengths[start] = 0

	while not queue.empty() and len(goals) > 0:

		cur = queue.get()

		if cur in goals:
			shortestDistances[cur] = lengths[cur]
			goals.remove(cur)

		if cur not in visited:
			neighbors = graph[cur][:]
			shuffle(neighbors)
			for neighbor in neighbors:
				if neighbor not in visited:
					queue.put(neighbor)
					if neighbor not in lengths.keys():
						lengths[neighbor] = lengths[cur] + 1

		visited.add(cur)

	return shortestDistances

def shortestPathToGoal(graph, start, goal):
	queue = q.Queue()
	queue.put(start)
	visited = set()
	paths = {}

	while not queue.empty():
		# print("Running search...")
		cur = queue.get()
		# print("Initialized cur...")

		if cur == goal:
			# print("Found Goal...")
			break
		# print("Checked if is goal...")
		if cur not in visited:
			# print("Before: " + str(graph[cur]))
			# graph[cur].sort(key=actually_visited.get)
			# print("After: " + str(graph[cur]))
			for neighbor in graph[cur]:
				# print("Cur's neighbors: ", cur, graph[cur])
				if neighbor not in visited:
					# print("Adding neighbor...")
					queue.put(neighbor)
					if neighbor not in paths.keys():
						paths[neighbor] = cur
					# print("Adding neighbor: ", neighbor, cur)

		visited.add(cur)
		# print("Visited: ", len(visited))

	return mapToStack(paths, goal, start)


def mapToStack(paths, goal, start):
	stack = deque()
	stack.append(goal)
	cur = goal
	# print(paths)
	while cur in paths.keys():
		cur = paths[cur]
		if cur == start:
			break
		stack.append(cur)

	return stack

def shortest_distances_to_goal(graph, visited, cur, goal, shortest_distances):
	if cur == goal:
		shortest_distances[cur] = 0
		return
	visited.add(cur)
	shortest_distances[cur] = len(graph.keys()) + 1
	for neighbor in graph[cur]:
		if neighbor not in visited:
			print("Checking neighbor " + str(neighbor) + " of " + str(cur))
			shortest_distances_to_goal(graph, visited, neighbor, goal, shortest_distances)
		shortest_distances[cur] = min(shortest_distances[cur], shortest_distances[neighbor] + 1)
	if shortest_distances[cur] == 26:
		print("Invalid distance: " + str(cur))
		for neighbor in graph[cur]:
			print(neighbor, shortest_distances[neighbor])
