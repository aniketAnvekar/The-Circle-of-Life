import AgentUtils as au

class Agent2:

    def __init__(self, graph, start, config):
        self.position = start
        self.graph = graph
        self.config = config
        self.total_pred_guess = 0
        self.total_pred_correct = 0
        self.total_prey_guess = 0
        self.total_prey_correct = 0
        self.visited = [0 for _ in self.graph.keys()]

    def update(self, predator, prey):
        return au.advanced_update_agent(self, predator, prey)