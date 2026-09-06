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

#function to convert unit square [0,1]x[0,1] to barycentric coordinates via Duffy transform
def uv2bc(u, v):
    u = toMpMath(u)
    v = toMpMath(v)

    #Duffy transform implemented below; defining a set of BC coordinates (A, B, C) from a point in the unit square (u, v)
    #The singular side occurs at u=1
    A = u
    B = v * (1-u)
    C = (1 - u) * (1 - v)
    return [A, B, C]

#function to convert barycentric coordinates to unit square coordinates (u, v) via Duffy transform
def bc2uv(x):
    x = [toMpmath(i) for i in x]
    u = x[0]

    if (u == 1):
        v = 0
        return [u, v]
    else:
        v = (x[1] / (1 - x[0]))
        return [u, v]

#function to calculate the dirichlet pdf at a given BC coordinate
def dirichletpdf(x, alpha):
    x = [toMpMath(i) for i in x]
    alpha = [toMpMath(j) for j in alpha]

    prod = 1 #for Dirichlet kernel \prod(x_1^alpha_i)
    a0 = 0 #for calculating beta function constant 
    g_prod = 1 #for calculating beta function constant
    area = toMpMath('0.5') * mp.sqrt(mp.mpf('3')) / 2 #barycentric triangle area

    for i in range(len(x)):
        try:
            prod *= (x[i]**(alpha[i][-1]))
        except ZeroDivisionError:
            prod = inf
        a0 += alpha[i]
        g_prod *= gm(alpha[i])
    g0 = gm(a0)
    beta_func = g0 / g_prod
    dirpdf = prod * beta_func * (mp.mpf('0.5') / area)

    return dirpdf