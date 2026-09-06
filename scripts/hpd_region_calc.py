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
    alphas = [toMpMath(alpha1), toMpMath(alpha2), toMpMath(alpha3)]

    resolution = 5
    resolution_ts = mp.mpf('0.001')

    dirichlet_object = Dirichlet_Distribution(resolution, resolution_ts, alphas)
    dirichlet_object.plotSimplex()
    dirichlet_object.integrateWithGQ()


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
            prod *= (x[i]**(alpha[i]-1))
        except ZeroDivisionError:
            prod = inf
        a0 += alpha[i]
        g_prod *= gm(alpha[i])
    g0 = gm(a0)
    beta_func = g0 / g_prod #beta function
    dirpdf = prod * beta_func * (mp.mpf('0.5') / area)

    return dirpdf

class Dirichlet_Distribution:

    #mesh properties
    vertices = [] #store BC vertices of each subtriangle, starting with (0,0,1), going right to left, bottom to top
    triangles = [] #store subtriangles by their vertex indices
    index_vert = [] #index for keeping track of vertices; index(0,0,1) = 0

    #this array stores relevant information about the mesh and distribution
    #the i-th entry will be [prob mass of subtriangle i, vertex indices of subtriangle, ...
        # ..., number of nodes (out of 13) in the HPD region, T/F on whether the subtriangle is split]
    sub_info = []

    #distribution parameters
    alphas = [0, 0, 0]

    #resolution of mesh, tanh-sinh step, domain area
    res = 0
    res_ts = 0
    area = mp.mpf('0.5') * (mp.sqrt(mp.mpf('3'))/2)

    '''
    Roots/nodes of the Dunavant 13-point (7-degree) rule. These are the simplex-specific barycentric coordinates for Gaussian quadrature, precalculated.
    The paper is called "HIGH DEGREE EFFICIENT SYMMETRICAL GAUSSIAN QUADRATURE RULES FOR THE TRIANGLE" by D. A. Dunavant.
    To use this, conceptualize the symmetric group S3; Each coordinate (node) is of the form [ralpha, rbeta, rgamma]. 
    The first column represents the identity permutation coordinate (1/3, 1/3, 1/3) in the center of the triangle.
    The second column represents three nodes, any permutation of (0.479, 0.260, 0.260).
    The third column, same thing, three nodes of some permutation of (0.86, 0.06, 0.06)
    The last column represents six nodes, as there are six different permutations of these three distinct barycentric coordinates. 
    Hence, these represent the 13 distinct nodes that will be sampled in every smooth subtriangle.
    '''
    ralpha = [toMpMath(1/3), mp.mpf('0.479308067841920'),  mp.mpf('0.869739794195568'),  mp.mpf('0.048690315425316')]
    rbeta = [toMpMath(1/3),  mp.mpf('0.260345966079040'),  mp.mpf('0.065130102902216'),  mp.mpf('0.312865496004874')]
    rgamma = [toMpMath(1/3), mp.mpf('0.260345966079040'), mp.mpf('0.065130102902216'), mp.mpf('0.638444188569810')]

    #weights at the nodes above. Each column from above will use the weight corresponding the the column below.
    #for example, the point (1/3, 1/3, 1/3) on each subtriangle will have the weight -0.14957...
    weights = [mp.mpf('-0.149570044467682'),  mp.mpf('0.175615257433208'),  mp.mpf('0.053347235608838'),  mp.mpf('0.077113760890257')]

    #mass of the domain, for accountability purposes after performing all calculations
    mass = 0

    #parameter for s, t-domain. These are the roughly-picked intervals on the real lines that go into our u-v unit square as a result of the tanh-sinh function.
    t_extrema = 2
    s_max = 3.64
    s_min = 2.4

    #stores subtriangle and its variance in an array for determining which triangles to use tah-sinh on.
    subtriangle_variance = []

    #iterator check: 0=first pass, 1=second pass, 2=3rd pass. This is for if we decide to do a subtriangle division after one 'pass', or pdf calculation across the domain.
    iterator = 0

    #constructor
    def __init__(self, a, h, params):

        self.res = a
        self.res_ts = h
        self.alphas = [toMpMath(i) for i in params]

        #segment sides of triangle for grid to perform numerical integration in barycentric coordinates
        n = mp.linspace(0, 1, a)

        grid = [n, n, n]

        #construct list of vertices from L->R (0,0,1) -> (0,1,0) and y=0 to y=sqrt(3)/2
        #construct list of vertex indices
        for i in range(len(n)):
            j = 0
            rowindex = []
            while((i+j) <= (a-1)):
                rowindex.append(len(self.vertices))
                self.vertices.append([n[i], n[j], n[a - i - j - 1]])
                j += 1
            self.index_vert.append(list(rowindex))

        #construct list of subtriangles, each characterized by three vertex indices
        for i in range(len(self.index_vert) - 1):
            for j in range(len(self.index_vert[i]) - 1):
                self.triangles.append([self.index_vert[i][j], self.index_vert[i][j+1], self.index_vert[i+1][j]])
                if((j+1) != (len(self.index_vert[i]) - 1)):
                    self.triangles.append([self.index_vert[i][j+1], self.index_vert[i+1][j], self.index_vert[i+1][j+1]])

    def getVertices(self):
        print(f"vertices: {self.vertices}")
        print(f"index of vertices: {self.index_vert}")
        print(f"triangles: {self.triangles}")

    def plotSimplex(self):
        plt.figure(1, figsize=(8,8))
        ax = plt.gca()

        #plot simplex/BC triangle
        triangle = mpatches.Polygon([[0,0], [1,0], [0.5, sqrt(3)/2]], closed=True, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(triangle)

        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, sqrt(3)/2 + 0.1)
        ax.set_aspect('equal', adjustable='box')

        #plot the vertices of each subtriangle
        for i in range(len(self.vertices)):
            x, y = bc2xy(self.vertices[i])
            plt.plot(x, y, 'k.')

        #for reference, plot the three vertices of the parent triangle and label them
        a, b = bc2xy([0, 0, 1])
        plt.plot(a, b, 'co', label='(0, 0, 1)')
        a, b = bc2xy([1, 0, 0])
        plt.plot(a, b, 'mo', label='(1, 0, 0)')
        a, b = bc2xy([0, 1, 0])
        plt.plot(a, b, 'go', label='(0, 1, 0)')
        plt.legend()
        o = f"Resolution: n={self.res}"
        plt.figtext(0.375, 0.075, o)

        plt.savefig("/home/jay/Documents/Research/CNRE/Dirichlet/hpd_calc_visual.png")

    #function for obtaining subtriangle area; used in the event AMR is employed and subtriangles are not uniform area
    def getSubArea(self, subtriangle):
        x_1, y_1 = bc2xy(self.vertices[subtriangle[1]])
        x_0, y_0 = bc2xy(self.vertices[subtriangle[0]])

        sidelength = mp.sqrt((x_1 - x_0)**2 + (y_1-y_0)**2)
        return self.area * sidelength**2

    #function for getting the local (within a subtriangle) barycentric coordinates given the parent BC triangle coords
    def localBCfromRefBC(self,subtriangle, bc):
        lambdas = [toMpMath(i) for i in bc]

        mu_0 = (lambdas[0] * self.vertices[subtriangle[0]][0]) + (lambdas[1] * self.vertices[subtriangle[1]][0])\
                + (lambdas[2] * self.vertices[subtriangle[2]][0]) 
        mu_1 = (lambdas[0] * self.vertices[subtriangle[0]][1]) + (lambdas[1] * self.vertices[subtriangle[1]][1])\
                + (lambdas[2] * self.vertices[subtriangle[2]][1])
        mu_2 = (lambdas[0] * self.vertices[subtriangle[0]][2]) + (lambdas[1] * self.vertices[subtriangle[1]][2])\
                + (lambdas[2] * self.vertices[subtriangle[2]][2])

        return [mu_0, mu_1, mu_2]

    #for a given subtriangle, return list of node locations in reference barycentric coordinates, and weights at the nodes
    def gaussQuadNodes(self, subtriangle):
        subtriangle_nodes = []

        #the nodes will be permutations of [ralpha[i], rbeta[i], rgamma[i]] with respect to the S3 symmetry group
        for i in range(4):

            #for i=0,1,2, get [ralpha[i],rbeta[i],rgamma[i]] and distinct permutations at each i
            w = [self.ralpha[i], self.rbeta[i], self.rgamma[i]]

            #if i=0, note there is only one distinct permutation of (1/3,1/3,1/3)
            if(i==0):
                subtriangle_nodes.append([self.localBCfromRefBC(subtriangle, w), self.weights[i]])
                continue
            else: #since there are three distinct permutations for each i=1,2
                for j in range(len(w)):
                    subtriangle_nodes.append([self.localBCfromRefBC(subtriangle, w), self.weights[i]])
                    w = w[-1:] + w[:-1] #np.roll them without the `np`

        #since the last col i=3 has six valid permutations, we need the final three nodes
        w = [self.ralpha[3], self.rgamma[3], self.rbeta[3]]
        for i in range(3):
            subtriangle_nodes.append([self.localBCfromRefBC(subtriangle, w), self.weights[3]])
            w = w[-1:] + w[:-1]
        
        return subtriangle_nodes

    #calculating Dirichlet pdf and weight at each node in the subtriangle
    #return subtriangle pdf mass via quadrature
    def gaussQuadSub(self, subtriangle):
        sub_nodes = self.gaussQuadNodes(subtriangle) 
        sub_value = 0
        cell_values = []

        #for each of the 13 nodes, calculate PDF there, multiply by quadrature weight
        for i in range(len(sub_nodes)):
            f_dirichlet = toMpMath(dirichletpdf(sub_nodes[i][0], self.alphas))
            value = f_dirichlet * toMpMath(sub_nodes[i][1])
            cell_values.append(value)
            sub_value += value
        
        #calculate PDF variance in cell for sorting
        cell_variance = max(cell_values) - min(cell_values)
        self.subtriangle_variance.append([subtriangle, cell_variance])

        return sub_value * self.getSubArea(subtriangle)

    #function to integrate PDF across the Dirichlet distribution domain
    def integrateWithGQ(self):
        integral_list = []
        total_mass = 0 #should equal 1 at the end

        #check if this is the first time integrating the domain
        if self.iterator > 0:
            self.subtriangle_variance = []

        print("\ncalculating probability masses...\n")

        #for each subtriangle in the domain, calculate the probability mass it contains
        for i in range(len(self.triangles)):
            sub_mass = toMpMath(self.gaussQuadSub(self.triangles[i]))

            #if this is NOT the first pass, check for any divided subtriangles from AMR
            if(self.iterator == 1):
                if(self.sub_info[i][3] == True):
                    self.sub_info[i] = [0, self.triangles[i], 0, True]
                    continue
                else:
                    self.sub_info[i] = [sub_mass, self.triangles[i], 0, False]

            #if this is the first pass, fill out sub_info
            elif(self.iterator == 0):
                self.sub_info.append([sub_mass, self.triangles[i], 0, False])

            #accumulate the probability mass now
            total_mass += sub_mass
            if i % 5000 == 0:
                print(f"{i}/{len(self.triangles)} subtriangle masses computed")

        ### Uncomment to verify total domain mass ###
        #print(f"total domain mass is {total_mass}")

            

if __name__ == "__main__":
    main()