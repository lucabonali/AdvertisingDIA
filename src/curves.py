import numpy as np

HIGH = 30
MEDIUM = 20
LOW = 10


def n(x):
    # The true function to estimate
    return (1.0 - np.exp(-5.0*x)) * 100


def fun(x):
    return (100 * (1.0 - np.exp(-4 * x + 3 * x ** 3))) / 5


def true(x):
    val = (1.0 - np.exp(9. - 1. * x)) * 30
    return np.array(list(map(lambda v: v if v >= 0.0 else 0.0, val.tolist())))


def true2(x):
    val = (1.0 - np.exp(0.5 - 0.5 * x)) * 10
    return np.array(list(map(lambda v: v if v >= 0.0 else 0.0, val.tolist())))


def true3(x):
    val = (1.0 - np.exp(0.2 - 0.5 * x)) * 50
    return np.array(list(map(lambda v: v if v >= 0.0 else 0.0, val.tolist())))


def generic(x, a=LOW, b=25, c=7):
    """
    Generic sigmoid function
    :param x: axis x value
    :param a: limit for x -> inf
    :param b: translation over axis x
    :param c: slope
    :return: sigmoid in x
    """
    return a / (1 + np.exp(b - (c * x)))

