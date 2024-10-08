class OptimizationAlgorithm:

    def __init__(self, *args, **kwargs):
        if 'name' in kwargs:
            self.name = kwargs['name']
        else:
            self.name = self.__class__.__name__

    def train(self, history):
        raise NotImplementedError("This method should be overridden by subclasses")

    def step(self):
        raise NotImplementedError("This method should be overridden by subclasses")
