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
    alpha1 = 7
    alpha2 = 7
    alpha3 = 7
    alphas = [toMpMath(alpha1), toMpMath(alpha2), toMpMath(alpha3)]

    resolution = 51
    resolution_ts = mp.mpf('0.001')
    p = mp.mpf('0.95')

    dirichlet_object = Dirichlet_Distribution(resolution, resolution_ts, alphas, p)
    dirichlet_object.plotSimplex()
    dirichlet_object.calcHPDArea()


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
    p = 0
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
    ts_bounds = [t_extrema, s_min, s_max]

    #stores subtriangle and its variance in an array for determining which triangles to use tah-sinh on.
    subtriangle_variance = []

    #iterator check: 0=first pass, 1=second pass, 2=3rd pass. This is for if we decide to do a subtriangle division after one 'pass', or pdf calculation across the domain.
    iterator = 0

    #constructor
    def __init__(self, a, h, params, p):

        self.res = a
        self.res_ts = toMpMath(h)
        self.p = toMpMath(p)
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


    #function to get threshold density t, determines subtriangles in or out of HPD region
    def getThresholdDensity(self):

        #bisection parameters
        M = 0
        R = 0
        L = 0

        #first, make sure subtriangle list is filled out
        self.integrateWithGQ()

        #if there exist parameters less than 1, we will need tanh-sinh integration
        if((self.alphas[0] < 1) or (self.alphas[1] < 1) or (self.alphas[2] < 1)):
            self.callTanhSinh()

        #find subtriangle with largest representative density and take this density as the upper bound R
        for i in range(len(self.sub_info)):

            #if subtriangle is divided from AMR, skip:
            if self.sub_info[i][3] == True:
                continue

            area_sub = self.getSubArea(self.sub_info[i][1])

            #if the iterated subtriangle has a density greater than the current R, make it the new R
            if ((1 / area_sub) * toMpMath(self.sub_info[i][0])) > R:
                R = toMpMath((1 / area_sub) * toMpMath(self.sub_info[i][0]))

        #iterate 20 times and perform bisection each time on the list of subtriangle densities
        for j in range(0, 20):
            M = toMpMath((L + R) / 2)
            g = 0

            #use midpoint M and calculate proportion of probability mass with subtriangles with densities above M
            for l in range(len(self.sub_info)):
                if(self.sub_info[i][3] == True):
                    continue
                area_sub = self.getSubArea(self.sub_info[i][1])
                if(((1 / area_sub) * toMpMath(self.sub_info[l][0])) > M): #if subtriangle rep. density > M...
                    g += toMpMath(self.sub_info[l][0]) #...accumulate the probability mass of the subtriangle

            #if the proportion of the probability mass g is greater than our target p, M too inclusive
            #M needs to be larger, take the new lower bound L to be the current M
            if(g > self.p):
                L = toMpMath(M)

            #if the proportion g is less than p, then M is too exclusive
            #M needs to be smaller, take new upper bound R to be the current M, bisect again
            elif(g < self.p):
                R = toMpMath(M)

            if(j % 5 == 0):
                print(f"{j}/{20} bisections performed")

        return M
            
    #function to classify boundary of HPD region
    def classifyBoundary(self):
        t = toMpMath(self.getThresholdDensity())
        print(f"\nboundary threshold density = {t}")
        print(f"classifying boundary...")

        #iterate through sub info list
        for i in range(len(self.sub_info)):
            node_threshold_count = 0
            if(self.sub_info[i][3] == True): #if the subtriangle is marked as divided via AMR
                continue

            ###### THE CODE HERE IS FOR WHAT TO DO IF WE ENCOUNTER A TANH-SINH TRIANGLE ######
            #####  
            elif(self.sub_info[i][2] == -1): #if the subtriangle mass was calculated with tanh-sinh...
                if((1/self.getSubArea(self.sub_info[i][1]) * self.sub_info[i][0]) > t): #pass if mass above t
                    continue
                elif((1/self.getSubArea(self.sub_info[i][1]) * self.sub_info[i][0]) < t): #treat as outside HPD otherwise
                    self.sub_info[i][2] = 0
                    continue
            #####

            ###### HERE WE CONSIDER ONLY GAUSSIAN SUBTRIANGLES #####
            #####
            sub_nodes = self.gaussQuadNodes(self.sub_info[i][1]) #get Gaussian nodes for subtriangle
            for j in range(len(sub_nodes)): #for each node in the subtriangle...
                f_dirichlet = toMpMath(dirichletpdf(sub_nodes[j][0], self.alphas)) #calculate Dirichlet pdf at subtriangle node
                if f_dirichlet > t: #if the Dirichlet pdf at the node > t, count it
                    node_threshold_count += 1
            self.sub_info[i][2] = node_threshold_count #denote number of nodes in subtriangle with PDF values above t
            if(i % 5000 == 0):
                print(f"{i}/{len(self.sub_info)} subtriangles classified")

    #function to calculate the area of the HPD region
    def calcHPDArea(self):

        ###### FRACTIONAL SUBTRIANGLES - SUM NORMALIZED WEIGHTS OF NODES ABOVE THRESHOLD INSTEAD OF i/13 ######

        areaHPD = 0
        massHPD = 0

        #update sub_info
        self.classifyBoundary()
        print("calculating HPD area...")

        for i in range(len(self.sub_info)):
            if(self.sub_info[i][3] == True): #if it is a subdivided subtriangle from AMR, skip
                continue
            elif(self.sub_info[i][2] == -1): # if the subtriangle is a tanh-sinh subtriangle within the HPD region
                areaHPD += self.getSubArea(self.sub_info[i][1])
                massHPD += toMpMath(self.sub_info[i][0])
            elif(self.sub_info[i][2] < 0) and (self.sub_info[i][2] != -1): #if the subtriangle is a boundary tanh-sinh...
                print("semi-tanh-sinh detected")
                #will return to work on this
                continue
            elif(self.sub_info[i][2] == 13): #if the subtriangle is a GQ subtriangle with all nodes inside HPD region
                areaHPD += self.getSubArea(self.sub_info[i][1])
                massHPD += toMpMath(self.sub_info[i][0])
            #update this calculation with the method suggested at the top of this function
            elif((self.sub_info[i][2] < 13) and (self.sub_info[i][2] > 0)): #if the subtriangle is GQ with partial nodes in HPD region
                areaHPD += toMpMath(self.sub_info[i][2] / 13) * self.getSubArea(self.sub_info[i][1])
                massHPD += toMpMath(self.sub_info[i][2] / 13) * toMpMath(self.sub_info[i][0])
            elif(self.sub_info[i][2] == 0): #if the subtriangle is excluded from the HPD region, skip
                continue
            
            #length of sub info may not be a good indicator if AMR is to be implemented extensively,
            if(i%5000 == 0):
                print(f"{i}/{len(self.sub_info)} of the simplex area calculated")

        print("\n***********************************************")
        print(f"*** for parameters {self.alphas}, HPD area is {areaHPD} with a probability mass of {massHPD}")
        print("*************************************************")
        return areaHPD

    #function to calculate the u, v bounds on which tanh-sinh integration is to be performed
    def getTanhSinhDomainBounds(self, subtriangle):
        alt_ref = mp.mpf('0.9931285991850950') #constant; if integrating in u, constant v, and vice versa
        alt_0 = toMpMath('0.5') * (alt_ref + 1) #shift from [-1, 1] to [0, 1]
        b1 = 0
        b2 = 0
        b3 = 0
        bounds = [b1, b2, b3] #bounds for the tanh-sinh domain

        #basically calculates the whole tanh sinh function given a specific node in the subtriangle
        def calcSumTerm(t, uv_var):
            uv = mp.mpf('0.5') * mp.tanh((mp.pi / 2) * mp.sinh(t)) + mp.mpf('0.5') #tanh sinh function here

            #if we are integrating in the u direction:
            if(uv_var == 'u'):
                f = toMpMath(dirichletpdf(self.localBCfromUV(subtriangle, uv, alt_0), self.alphas))
                u_jacobian = uv #the jacobian (1-u) will change if v is constant
            elif(uv_var == 'v'):
                f = toMpMath(dirichletpdf(self.localBCfromUV(subtriangle, alt_0, uv), self.alphas))
                u_jacobian = alt_0 #the jacobian 1-u is constant because u is constant
            else:
                print("variable not u or v")
                quit()
            w_k = (mp.pi / 4) * (mp.cosh(t) / ((mp.cosh((mp.pi/2) * mp.sinh(t)))**2)) #tanh-sinh weight, the deriv of the tanh sinh function
            return f * w_k * self.res_ts**2 * (1 - u_jacobian) * 2 * self.getSubArea(subtriangle)

        #for u->1, v->0, v->1
        for i in range(len(self.ts_bounds)):
            k = 0
            if(i==0):
                var = 'u'
            else:
                var = 'v'
            while True:
                bounds[i] = self.ts_bounds[i] + (k * self.res_ts) #increment by k*h
                sum_term = calcSumTerm(bounds[i], var) #calculate TS function at the bound

                if (sum_term < 1e-40) and (k > 0): #if bound within threshold, take it as the true bound
                    break
                elif(((sum_term == inf) and (k > 0)) or (mp.isnan(sum_term) == True)): #if bound is nan or inf, back up one h, take it as true bound
                    bounds[i] -= self.res_ts
                    break
                else: #if we can push bound further, do so by incrementing
                    if(i==1):
                        k -= 1
                    else:
                        k += 1
        
        return [[-bounds[0], bounds[0]], [bounds[1], bounds[2]]]

    #function for calculating subtriangle barycentric coordinates from U, V coordinates
    #add orientation functionality
    def localBCfromUV(self, subtriangle, u, v):
        u = toMpMath(u)
        v = toMpmath(v)
        lambdas = uv2bc(u, v)

        coords = self.localBCfromRefBC(subtriangle, lambdas)
        return coords


if __name__ == "__main__":
    main()