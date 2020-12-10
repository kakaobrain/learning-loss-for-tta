import copy


class EMA:
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def state_dict(self):
        return copy.deepcopy(self.shadow)

    def __len__(self):
        return len(self.shadow)

    def __call__(self, module, step=None):
        if step is None:
            mu = self.mu
        else:
            # see : https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/ExponentialMovingAverage?hl=PL
            mu = min(self.mu, (1. + step) / (10 + step))

        for name, x in module.state_dict().items():
            if name in self.shadow:
                new_average = (1.0 - mu) * x + mu * self.shadow[name]
                self.shadow[name] = new_average.clone()
            else:
                self.shadow[name] = x.clone()
