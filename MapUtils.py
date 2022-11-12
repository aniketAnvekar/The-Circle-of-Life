import queue as q
from collections import deque
from random import shuffle
import Predator as pr


def get_shortest_distances_to_goals(graph, start, goals):
    queue = q.Queue()
    queue.put(start)
    visited = set()
    shortest_distances = {}
    lengths = {start: 0}

    while not queue.empty() and len(goals) > 0:

        cur = queue.get()

        if cur in goals:
            shortest_distances[cur] = lengths[cur]
            goals.remove(cur)

        if cur not in visited:
            neighbors = graph[cur][:]
            # shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.put(neighbor)
                    if neighbor not in lengths.keys():
                        lengths[neighbor] = lengths[cur] + 1

        visited.add(cur)

    return shortest_distances


def shortest_path_to_goal(graph, start, goal):
    queue = q.Queue()
    queue.put(start)
    visited = set()
    paths = {}

    while not queue.empty():
        cur = queue.get()
        if cur == goal:
            break
        if cur not in visited:
            for neighbor in graph[cur]:
                if neighbor not in visited:
                    queue.put(neighbor)
                    if neighbor not in paths.keys():
                        paths[neighbor] = cur

        visited.add(cur)

    return map_to_stack(paths, goal, start)


def map_to_stack(paths, goal, start):
    stack = deque()
    stack.append(goal)
    cur = goal
    while cur in paths.keys():
        cur = paths[cur]
        if cur == start:
            break
        stack.append(cur)

    return stack


def recursive_search(agent, depth, cur, sim_predator, prey_pos, visited):
    if cur == prey_pos:
        return -1 * depth

    visited.add(cur)
    sim_predator.update(cur)

    if cur == sim_predator.position:
        return None
    elif depth == 0:
        return get_shortest_distances_to_goals(agent.graph, cur, [prey_pos])[prey_pos]

    min_dist = 10000
    for neighbor in agent.graph[cur]:
        if neighbor not in visited:
            new_sim_predator = pr.Predator(agent.graph, agent.config, agent.position, simulation=sim_predator.position)
            dist = recursive_search(agent, depth - 1, neighbor, new_sim_predator, prey_pos, visited)
            if dist is not None:
                min_dist = min(min_dist, dist)

    if min_dist == 10000:
        min_dist = None

    return min_dist
