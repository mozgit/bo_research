class ProtoDomain:
    def __init__(self):
        self.n_dimensions = None
        self.bounds = None

    def is_within_domain(self, x):
        raise NotImplementedError("This method should be overridden by subclasses")

    def sample(self, n_samples = 1):
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_inequality_constraints(self):
        return []

    def get_equality_constraints(self):
        return []

    def generate_choices(self):
        return [[]]

    def get_dimensionality(self):
        return self.n_dimensions

    def encode(self, x):
        return x

    def decode(self, x):
        return x