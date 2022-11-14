from random import choice
import MapUtils as mp

import AgentUtils as au


class Agent10B:
    def __init__(self, graph, start, config):
        self.position = start
        self.graph = graph
        self.config = config
        self.total_pred_guess = 0
        self.total_pred_correct = 0
        self.total_prey_guess = 0
        self.total_prey_correct = 0
        self.visited = [0 for _ in self.graph.keys()]

        # initialization

        self.predator_q = [0 for _ in range(self.config["GRAPH_SIZE"])]
        self.first_run = True

    def update(self, predator, prey):

        if self.first_run:
            # finish initialization...
            self.predator_q[predator.position] = 1
            self.first_run = False

        ret = 0
        estimated_predator_position = au.pick_most_probable_spot(self, self.predator_q)
        options = list(
            filter(lambda x: self.predator_q[x] == self.predator_q[estimated_predator_position], self.graph.keys()))
        shortest = self.position - estimated_predator_position if self.position > estimated_predator_position else self.position + 50 - estimated_predator_position
        if len(options) != 1:
            distances = mp.get_shortest_distances_to_goals(self.graph, predator.position, options)
            shortest = max(distances.values())
            estimated_predator_position = choice([i for i in distances.keys() if distances[i] == shortest])

        if max(self.predator_q) < self.config["THRESHOLD"] and shortest > 2:
            estimated_predator_position = au.survey_partial_pred(self, predator)
        else:
            ret = au.advanced_update_agent(self, predator, prey,
                                           estimated_predator_position=estimated_predator_position)
        if ret == 0:
            au.general_move_agent(self)
        return ret