import AgentUtils as au


class Agent1:
    def __init__(self, graph, start, config):
        self.position = start
        self.graph = graph
        self.config = config

    def update(self, predator, prey):
        return au.basic_update_agent(self, predator, prey)
