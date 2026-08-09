"""
Environment wrappers and CybORG single-step adapters.
Includes compatibility patch for Gym RNG deepcopy in Python 3.10 / NumPy 1.26+.
"""

import copy
try:
    from gym.utils import seeding

    def _fixed_generator_ctor(bit_generator="MT19937", *args):
        if isinstance(bit_generator, str):
            from numpy.random._pickle import BitGenerators
            bit_gen_cls = BitGenerators[bit_generator]
            return seeding.RandomNumberGenerator(bit_gen_cls())
        else:
            return seeding.RandomNumberGenerator(bit_generator)

    seeding.RandomNumberGenerator._generator_ctor = _fixed_generator_ctor
except ImportError:
    pass
