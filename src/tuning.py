"""Explicit deterministic grid; no data leaks from the held-out test set."""
from itertools import product


def grid_configs(space):
    names = list(space)
    for combination in product(*(space[name] for name in names)):
        yield dict(zip(names, combination))
