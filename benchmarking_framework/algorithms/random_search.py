from .base import OptimizationAlgorithm

class RandomSearch(OptimizationAlgorithm):
    def __init__(self, domain, name = "Random Search"):
        super().__init__()
        self.name = name
        self.domain = domain

    def train(self, history):
        return self

    def step(self):
        return self.domain.sample()
