import numpy as np

from flatland.core.policy import Policy


class RandomPolicy(Policy):
    def __init__(self,
                 action_size: int = 5):
        super(RandomPolicy, self).__init__()
        self.action_size = action_size

    def act(self, handle: int, state, eps=0.):
        return np.random.choice(self.action_size)
