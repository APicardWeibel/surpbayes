"""
Exceptions for probability module
"""


class RenormError(Exception):
    """Exception when trying to compute Gaussian distribution from unnormalisable quadratic form"""


class NegativeKL(Exception):
    """Exception when KL computation yields negative value"""
