import MapUtils as mp
from random import choice
import numpy as np
import AgentUtils as au


class Agent5:

    def __init__(self, graph, start, config):
        self.position = start
        self.graph = graph
        self.config = config

        # initialization

        self.predator_q = [0 for _ in range(self.config["GRAPH_SIZE"])]
        self.first_run = True

    def update(self, predator, prey):
        if self.first_run:
            # finish initialization...
            self.predator_q[predator.position] = 1
            self.first_run = False

        estimated_predator_position = au.survey_partial_pred(self, predator)

        ret = au.basic_update_agent(self, predator, prey, estimated_predator_position=estimated_predator_position)

        if ret == 0:
            au.general_move_agent(self)

        return ret