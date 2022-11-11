import AgentUtils as au
import MapUtils as mp
import Predator as pr

class Agent2:

    def __init__(self, graph, start, config):
        self.position = start
        self.graph = graph
        self.config = config
        self.visited = [0 for x in self.graph.keys()]

    def update(self, predator, prey):
        return au.advanced_update_agent(self, predator, prey)


    #
    # def update(self, predator, prey):
    #
    #     min_neighbor = -1
    #     min_dist = 1000
    #     for neighbor in self.graph[self.position]:
    #
    #         sim_predator = pr.Predator(self.graph, self.config, self.position, simulation=predator.position)
    #         dist = self.function(4, neighbor, sim_predator, prey, set())
    #
    #         if not dist is None:
    #             if dist < min_dist or (dist == min_dist and self.visited[neighbor] <= self.visited[min_neighbor]):
    #                 min_dist = dist
    #                 min_neighbor = neighbor
    #
    #     if min_neighbor != -1:
    #         self.position = min_neighbor
    #     self.visited[self.position] = self.visited[self.position] + 1
    #
    #     return 1 if self.position == prey.position else -1 if self.position == predator.position else 0
    #
    # def function(self, depth, cur, sim_predator, prey, visited):
    #
    #     if cur == prey.position:
    #         return -1*depth
    #
    #     visited.add(cur)
    #     sim_predator.update(cur)
    #
    #     if cur == sim_predator.position:
    #         return None
    #     elif depth == 0:
    #         return mp.getShortestDistancesToGoals(self.graph, cur, [prey.position])[prey.position]
    #
    #     min_dist = 10000
    #     for neighbor in self.graph[cur]:
    #         if neighbor not in visited:
    #             new_sim_predator = pr.Predator(self.graph, self.config, self.position, simulation=sim_predator.position)
    #             dist = self.function(depth - 1, neighbor, new_sim_predator, prey, visited)
    #             if not dist is None:
    #                 min_dist = min(min_dist, dist)
    #
    #     if min_dist == 10000:
    #         min_dist = None
    #
    #     return min_dist
