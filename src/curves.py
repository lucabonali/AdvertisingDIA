import numpy as np

HIGH = 30
MEDIUM = 20
LOW = 10


def n(x):
    # The true function to estimate
    return (1.0 - np.exp(-5.0*x)) * 100


def fun(x):
    return (100 * (1.0 - np.exp(-4 * x + 3 * x ** 3))) / 5


def true1(x):
    val = (1.0 - np.exp(3. - 1. * x)) * 15
    return np.array(list(map(lambda v: v if v >= 0.0 else 0.0, val.tolist())))


def true2(x):
    val = (1.0 - np.exp(0.5 - 0.1 * x)) * 10
    return np.array(list(map(lambda v: v if v >= 0.0 else 0.0, val.tolist())))


def true3(x):
    val = (1.0 - np.exp(20. - 1. * x)) * 40
    return np.array(list(map(lambda v: v if v >= 0.0 else 0.0, val.tolist())))


def true4(x):
    val = (1.0 - np.exp(0.08 - 0.05 * x)) * 10
    return np.array(list(map(lambda v: v if v >= 0.0 else 0.0, val.tolist())))


def true5(x):
    val = (1.0 - np.exp(0.1 - 0.2 * x)) * 20
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

