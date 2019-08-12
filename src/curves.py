import numpy as np

HIGH = 90
MEDIUM = 60
LOW = 40

p_c1, p_c2, p_c3 = 0.5, 0.4, 0.1


def n(x):
    # The true function to estimate
    return (1.0 - np.exp(-5.0*x)) * 100


def fun(x):
    return (100 * (1.0 - np.exp(-4 * x + 3 * x ** 3))) / 5


def filter0(val):
    return np.array(list(map(lambda v: v if v >= 0.0 else 0.0, val.tolist())))


# START: Google channel classes

def google_c1(x):
    return filter0(1.0 - np.exp(3. - 1. * x)) * HIGH


def google_c2(x):
    return filter0(1.0 - np.exp(0.4 - 0.15 * x)) * MEDIUM


def google_c3(x):
    return filter0(LOW / (1 + np.exp(5. - 0.5 * x)))


def google_agg(x):
    return (google_c1(x) * p_c1) + (google_c2(x) * p_c2) + (google_c3(x) * p_c3)

# END: Google channel classes


def true1(x):
    val = (1.0 - np.exp(3. - 2. * x)) * 45
    return np.array(list(map(lambda v: v if v >= 0.0 else 0.0, val.tolist())))


def true2(x):
    val = (1.0 - np.exp(0.5 - 0.4 * x)) * 70
    return np.array(list(map(lambda v: v if v >= 0.0 else 0.0, val.tolist())))


def true3(x):
    val = (1.0 - np.exp(2. - 1. * x)) * 50
    return np.array(list(map(lambda v: v if v >= 0.0 else 0.0, val.tolist())))


def true4(x):
    val = (1.0 - np.exp(0.08 - 0.5 * x)) * 35
    return np.array(list(map(lambda v: v if v >= 0.0 else 0.0, val.tolist())))


def true5(x):
    val = (1.0 - np.exp(0.6 - 0.5 * x)) * 40
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

