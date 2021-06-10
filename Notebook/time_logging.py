"""
Module with a decorator to print how long a function takes.

Author: Gabriel Peery
Date: 6/7/2021
"""
from functools import wraps
from time import perf_counter
from typing import Callable


def print_time(func : Callable):
    """Decorator to print the time of how long a function takes."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        print(f"{start:.3f} : '{func.__name__:>32}' starting")
        ret = func(*args, **kwargs)
        end = perf_counter()
        print(f"{end:.3f} : '{func.__name__:>32}'"
              f" took {end - start:.6f} seconds")
        return ret
    return wrapper
