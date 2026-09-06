# libraries here

from math import *
from math import factorial
from mpmath import gamma as gm
from mpmath import factorial as fac
import mpmath as mp
from mpmath import quad
import numpy as np
from scipy import optimize
from scipy import stats
from scipy.integrate import dblquad
from scipy.stats import dirichlet
from scipy.special import gamma, factorial, beta
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import patches as mpatches
import matplotlib.tri as tri

#set mpmath global precision for tanh-sinh quadrature
mp.mp.dps = 35

def main():

    #alpha params for dirichlet distribution here
    alpha1 = 1
    alpha2 = 2
    alpha3 = 3
    alphas = []


#function for ensuring all floating point numbers are in mpmath precision
def toMpMath(x):
    #if the arg is already a mpmath number, pass
    if isinstance(x, mp.mpf):
        return x
    #else, return it as a mpmath float
    else:
        return mp.mpf(str(x))

#function for converting barycentric to cartesian coordinates
def bc2xy(xyz):
    x = (xyz[0] * mp.mpf('0.5')) + (xyz[1] * 0) + (xyz[2] * 1)
    y = (xyz[0] * (mp.sqrt(mp.mpf('3')) / 2)) + (xyz[1] * 0) + (xyz[2] * 0)
    return [x,y]