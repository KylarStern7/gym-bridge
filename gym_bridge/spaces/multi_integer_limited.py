from gym import Space
import numpy as np


class MultiIntegerLimited(Space):
    """
    An [n_min, n_max] length list of integers from range [low_limit, high_limit].
    The argument to MultiIntegerLimited defines n_min, n_max and limits (minimum and maximum valid integer).

    Example Usage:

    >> self.observation_space = spaces.MultiIntegerLimited(0, 5, 0, 2)
    >> self.observation_space.sample()
        array([1,0,2,0], dtype=int8)
    """

    def __init__(self, n_min=0, n_max=52, low_limit=0, high_limit=51):
        """
        Initialize space.

        Args:
            n_min (int): minimum size of space
            n_max (int): maximum size of space
            low_limit (int): minimal valid integer
            high_limit (int): maximal valid integer
        """
        self.n_min = n_min
        self.n_max = n_max
        self.low_limit = low_limit
        self.high_limit = high_limit
        super(MultiIntegerLimited, self).__init__(None, np.int8)

    def sample(self):
        """Randomly sample an element of this space. Can be
        uniform or non-uniform sampling based on boundedness of space."""
        sample = np.random.choice(range(self.low_limit, self.high_limit+1), np.random.choice(range(self.n_min, self.n_max)))
        return sample

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        return ((self.low_limit <= x) & (x < self.high_limit+1)).all() and self.n_min <= len(x) <= self.n_max

    def to_jsonable(self, sample_n):
        """Convert a batch of samples from this space to a JSONable data type."""
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        """Convert a JSONable data type to a batch of samples from this space."""
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        return f'MultiIntegerLimited({self.n_min, self.n_max, self.low_limit, self.high_limit})'

    def __eq__(self, other):
        return isinstance(other, MultiIntegerLimited) and (self.n_min == other.n_min
                                                           and self.n_max == other.n_max
                                                           and self.low_limit == other.low_limit
                                                           and self.high_limit == other.high_limit)
