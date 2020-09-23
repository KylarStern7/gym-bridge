from gym import Space
import numpy as np


class MultiBinaryLimited(Space):
    '''
    An n-dimensional binary space with limited number of ones.
    The argument to MultiBinaryLimited defines n and limits (minimum and maximum numbers of ones).

    Example Usage:

    >> self.observation_space = spaces.MultiBinaryLimited(5, 0, 2)
    >> self.observation_space.sample()
        array([0,1,0,1,0], dtype=int8)
    '''

    def __init__(self, n, low_limit=0, high_limit=1):
        self.n = n
        self.low_limit = low_limit
        self.high_limit = high_limit
        super(MultiBinaryLimited, self).__init__((self.n,), np.int8)

    def sample(self):
        sample = np.zeros(self.n, dtype=self.dtype)
        sample[self.np_random.choice(self.n, self.np_random.random_integers(low=self.low_limit, high=self.high_limit),
                                     replace=False)] = 1
        return sample

    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        return ((x == 0) | (x == 1)).all() and self.low_limit <= np.count_nonzero(x) <= self.high_limit

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        return f'MultiBinaryLimited({self.n, self.low_limit, self.high_limit})'

    def __eq__(self, other):
        return isinstance(other, MultiBinaryLimited) and self.n == other.n and (self.low_limit == other.low_limit
                                                                                and self.high_limit == other.high_limit)
