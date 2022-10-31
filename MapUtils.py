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
			shuffle(graph[cur])
			for neighbor in graph[cur]:
				if neighbor not in visited:
					queue.put(neighbor)
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
			shuffle(graph[cur])
			for neighbor in graph[cur]:
				# print("Cur's neighbors: ", cur, graph[cur])
				if neighbor not in visited:
					# print("Adding neighbor...")
					queue.put(neighbor)
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


