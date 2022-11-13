import AgentUtils as au


class Agent3:
    def __init__(self, graph, start, config):
        self.position = start
        self.graph = graph
        self.config = config
        self.total_prey_guess = 0
        self.total_prey_correct = 0
        self.total_pred_guess = 0
        self.total_pred_correct = 0

        # initialization

        self.prey_q = [1 / (self.config["GRAPH_SIZE"] - 1) for _ in range(self.config["GRAPH_SIZE"])]
        self.prey_q[start] = 0
        self.found_prey = False

        # set up transition matrix

        self.prey_P = [[0 for _ in range(self.config["GRAPH_SIZE"])] for _ in
                       range(self.config["GRAPH_SIZE"])]  # transition probabilities from j to i
        for i in graph.keys():
            self.prey_P[i][i] = 1 / (len(self.graph[i]) + 1)
            for j in graph[i]:
                self.prey_P[i][j] = 1 / (len(self.graph[j]) + 1)

    def update(self, predator, prey):
        estimated_prey_position = au.survey_partial_prey(self, prey)

        ret = au.basic_update_agent(self, predator, prey, estimated_prey_position=estimated_prey_position)
        if ret == 0:
            au.general_move_agent(self)

        return ret
