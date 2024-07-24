from .base import OptimizationAlgorithm

class RandomSearch(OptimizationAlgorithm):
    def __init__(self, name = "Random Search"):
        super().__init__()
        self.name = name

    def train(self, history):
        return self

    def step(self, domain):
        return domain.sample()
