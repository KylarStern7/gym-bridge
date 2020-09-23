from gym import Space
import numpy as np


class MultiIntegerLimited(Space):
    '''
    An n-dimensional binary space with limited number of ones.
    The argument to MultiBinaryLimited defines n and limits (minimum and maximum numbers of ones).

    Example Usage:

    >> self.observation_space = spaces.MultiBinaryLimited(5, 0, 2)
    >> self.observation_space.sample()
        array([0,1,0,1,0], dtype=int8)
    '''

    def __init__(self, n_min=0, n_max=52, low_limit=0, high_limit=51):
        self.n_min = n_min
        self.n_max = n_max
        self.low_limit = low_limit
        self.high_limit = high_limit
        super(MultiIntegerLimited, self).__init__(None, np.int8)

    def sample(self):
        sample = np.random.choice(range(self.low_limit, self.high_limit+1), np.random.choice(range(self.n_min, self.n_max)))
        return sample

    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        return ((self.low_limit <= x) & (x < self.high_limit+1)).all() and self.n_min <= len(x) <= self.n_max

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        return f'MultiIntegerLimited({self.n_min, self.n_max, self.low_limit, self.high_limit})'

    def __eq__(self, other):
        return isinstance(other, MultiIntegerLimited) and (self.n_min == other.n_min
                                                           and self.n_max == other.n_max
                                                           and self.low_limit == other.low_limit
                                                           and self.high_limit == other.high_limit)
