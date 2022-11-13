import AgentUtils as au

class Agent6:
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

        estimated_predator_position = au.survey_partial_pred(self, predator)

        ret = au.advanced_update_agent(self, predator, prey, estimated_predator_position=estimated_predator_position)
        if ret == 0:
            au.general_move_agent(self)
        return ret