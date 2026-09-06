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