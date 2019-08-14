import numpy as np

HIGH = 90
MEDIUM = 60
LOW = 40
SHIFT = 10
p_c1, p_c2, p_c3 = 0.65, 0.305, 0.045


def n(x):
    # The true function to estimate
    return (1.0 - np.exp(-5.0*x)) * 100


def fun(x):
    return (100 * (1.0 - np.exp(-4 * x + 3 * x ** 3))) / 5


def filter0(val):
    return np.array(list(map(lambda v: v if v >= 0.0 else 0.0, val.tolist())))


""" START: Google channel classes """

def google_c1(x):
   return filter0(1.0 - np.exp(2. - 1. * x)) * HIGH


def google_c2(x):
   return filter0(1.0 - np.exp(0.4 - 0.15 * x)) * MEDIUM


def google_c3(x):
   return filter0(LOW / (1 + np.exp(5. - 0.5 * x)))


def google_agg(x):
   return (google_c1(x) * p_c1) + (google_c2(x) * p_c2) + (google_c3(x) * p_c3)


""" END: Google channel classes """

""" START: Facebook channel classes """

def facebook_c1(x):
    return filter0((LOW + SHIFT) / (1 + np.exp(5. - 0.7 * x)))

def facebook_c2(x):
    return filter0(MEDIUM / (1 + np.exp(5. - 0.7 * x)))

def facebook_c3(x):
    return filter0((LOW + SHIFT) / (1 + np.exp(15. - 1. * x)))

def facebook_agg(x):
    return (facebook_c1(x) * p_c1) + (facebook_c2(x) * p_c2) + (facebook_c3(x) * p_c3)

""" END: Facebook channel classes """

""" START: Instagram channel classes """

def instagram_c1(x):
    return filter0(HIGH / (1 + np.exp(6. - 1.3 * x)))

def instagram_c2(x):
    return filter0(MEDIUM / (1 + np.exp(12. - 1.3 * x)))

def instagram_c3(x):
    return filter0((LOW - 3*SHIFT) / (1 + np.exp(15. - 1.3 * x)))

def instagram_agg(x):
    return (instagram_c1(x) * p_c1) + (instagram_c2(x) * p_c2) + (instagram_c3(x) * p_c3)

""" END: Instagram channel classes """

""" START: Youtube channel classes """

def youtube_c1(x):
    return filter0((MEDIUM) / (1 + np.exp(4. - 0.3 * x)))

def youtube_c2(x):
    return filter0(LOW / (1 + np.exp(5. - 0.35 * x)))

def youtube_c3(x):
    return filter0((LOW - SHIFT) / (1 + np.exp(6. - 0.35 * x)))

def youtube_agg(x):
    return (youtube_c1(x) * p_c1) + (youtube_c2(x) * p_c2) + (youtube_c3(x) * p_c3)

""" END: Youtube channel classes """

""" START: Bing channel classes """

def bing_c1(x):
    return filter0(1.0 - np.exp(4. - 1. * x)) * MEDIUM

def bing_c2(x):
    return filter0((MEDIUM) / (1 + np.exp(7. - 0.8 * x)))

def bing_c3(x):
    return filter0((LOW) / (1 + np.exp(6. - 0.6 * x)))

def bing_agg(x):
    return (bing_c1(x) * p_c1) + (bing_c2(x) * p_c2) + (bing_c3(x) * p_c3)

""" END: Bing channel classes """


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

