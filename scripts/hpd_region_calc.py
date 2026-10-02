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
    alpha1 = 0.5
    alpha2 = 0.5
    alpha3 = 9
    alphas = (toMpMath(alpha1), toMpMath(alpha2), toMpMath(alpha3))

    resolution = 51
    resolution_ts = mp.mpf('0.1')
    p = mp.mpf('0.95')

    dirichlet_object = Dirichlet_Distribution(resolution, resolution_ts, alphas, p)
    dirichlet_object.plotSimplex()
    dirichlet_object.testTSBounds()
    dirichlet_object.calcHPDArea()
    dirichlet_object.getTanhSinhComparison()
    
    dirichlet_object.plotHPDRegion()


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
    x = [toMpMath(i) for i in x]
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

    #these arrays store relevant information about the mesh and distribution
    p_mass = [] #store probability mass by index of each subtriangle
    hpd_nodes = [] #store number of nodes contained within the HPD region
    subdivided = [] #store T/F on whether the subtriangle is divided via AMR

    #distribution parameters
    alphas = [0, 0, 0]

    #resolution of mesh, tanh-sinh step, domain area, total domain mass
    res = 0
    res_ts = 0
    p = 0
    area = mp.mpf('0.5') * (mp.sqrt(mp.mpf('3'))/2)
    total_mass = 0

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
    ralpha = (toMpMath(1/3), mp.mpf('0.479308067841920'),  mp.mpf('0.869739794195568'),  mp.mpf('0.048690315425316'))
    rbeta = (toMpMath(1/3),  mp.mpf('0.260345966079040'),  mp.mpf('0.065130102902216'),  mp.mpf('0.312865496004874'))
    rgamma = (toMpMath(1/3), mp.mpf('0.260345966079040'), mp.mpf('0.065130102902216'), mp.mpf('0.638444188569810'))

    #weights at the nodes above. Each column from above will use the weight corresponding the the column below.
    #for example, the point (1/3, 1/3, 1/3) on each subtriangle will have the weight -0.14957...
    weights = (mp.mpf('-0.149570044467682'),  mp.mpf('0.175615257433208'),  mp.mpf('0.053347235608838'),  mp.mpf('0.077113760890257'))

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
    length_record = 0 #value for keeping track of already-computed subtriangles

    #constructor
    def __init__(self, a, h, params, p):

        self.res = a
        self.res_ts = toMpMath(h)
        self.p = toMpMath(p)
        self.alphas = [toMpMath(i) for i in params]

        #segment sides of triangle for grid to perform numerical integration in barycentric coordinates
        n = mp.linspace(0, 1, a)

        #construct list of vertices from L->R (0,0,1) -> (0,1,0) and y=0 to y=sqrt(3)/2
        #construct list of vertex indices
        for i in range(len(n)):
            j = 0
            rowindex = []
            while((i+j) <= (a-1)):
                rowindex.append(len(self.vertices))
                self.vertices.append((n[i], n[j], n[a - i - j - 1])) #
                j += 1
            self.index_vert.append(list(rowindex))

        '''
        self.vertices = np.array(self.vertices, dtype=object)
        self.index_vert = np.array(self.index_vert, dtype=object)
        '''

        #avoid append, use numpy functions for large arrays


        #construct list of subtriangles, each characterized by three vertex indices
        for i in range(len(self.index_vert) - 1):
            for j in range(len(self.index_vert[i]) - 1):
                self.triangles.append([self.index_vert[i][j], self.index_vert[i][j+1], self.index_vert[i+1][j]])
                if((j+1) != (len(self.index_vert[i]) - 1)):
                    self.triangles.append([self.index_vert[i][j+1], self.index_vert[i+1][j], self.index_vert[i+1][j+1]])

        #self.triangles = np.array(self.triangles, dtype=object)

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

        #plot the mesh if uncommented
        '''
        for i in range(len(self.triangles)):
            x_list = []
            y_list = []
            for j in range(3):
                x, y = bc2xy(self.vertices[self.triangles[i][j]])
                x_list.append(x)
                y_list.append(y)
            x, y = bc2xy(self.vertices[self.triangles[i][0]])
            x_list.append(x)
            y_list.append(y)
            plt.plot(x_list, y_list, 'r-')
        '''
        


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

        #plt.show()
        plt.savefig("C://Users//T Mill//Documents//Research//CNRE Summer 2026//Dirichlet_uq_eigenspace//hpd_calc_visuals.png")
        #plt.savefig("/home/jay/Documents/Research/CNRE/Dirichlet/hpd_calc_visual.png")

    def plotHPDRegion(self):
        self.plotSimplex()
        for i in range(len(self.triangles)):
            x_tri = []
            y_tri = []
            if(self.hpd_nodes[i] > 0):
                for j in range(3):
                    x, y = bc2xy(self.vertices[self.triangles[i][j]])
                    x_tri.append(x)
                    y_tri.append(y)
            plt.plot(x_tri, y_tri, 'r.')
        #plt.savefig("/home/jay/Documents/Research/CNRE/Dirichlet/hpd_calc_visual.png")
        plt.savefig("C://Users//T Mill//Documents//Research//CNRE Summer 2026//Dirichlet_uq_eigenspace//hpd_calc_visuals.png")

    def getTanhSinhComparison(self):
        i = np.argmax(self.p_mass)
        print(f"subtriangle {self.triangles[i]} has p mass {self.p_mass[i]}")
        

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

        return sub_value * self.getSubArea(subtriangle)

    #function to integrate PDF across the Dirichlet distribution domain
    def integrateWithGQ(self): #potentially make this function only recalculate subdivided triangles for degenerate cases
        #total_mass = self.total_mass #should equal 1 at the end

        print("\ncalculating probability masses...\n")
        #for each subtriangle in the domain, calculate the probability mass it contains
        '''
        for i in range(len(self.triangles)):

            #if this is NOT the first pass, check for any divided subtriangles from AMR
            if(self.iterator == 1):
                if(self.subdivided[i] == True) or (self.subdivided[i] == 1): #set parent subtriangle properties to zero
                    self.p_mass[i] = 0
                    self.hpd_nodes[i] = 0
                    continue
                elif(self.subdivided[i] == False) or (self.subdivided[i] == 0):# and (i >= 1521):
                    sub_mass = toMpMath(self.gaussQuadSub(self.triangles[i]))
                    self.p_mass[i] = sub_mass
                    self.hpd_nodes[i] = 0 #nodes in HPD will come later
                else:
                    continue
            

            #if this is the first pass, fill out p_mass, hpd_nodes, subdivided
            elif(self.iterator == 0):
                sub_mass = toMpMath(self.gaussQuadSub(self.triangles[i]))
                self.p_mass.append(sub_mass)
                self.hpd_nodes.append(0) #will be calculated after boundary classification
                self.subdivided.append(False)

            #accumulate the probability mass now
            total_mass += self.p_mass[i]
            if i % 5000 == 0:
                print(f"{i}/{len(self.triangles)} subtriangle masses computed")
        '''
        if(self.iterator == 1):
            for i in range(self.length_record, len(self.triangles)):
                if(self.subdivided[i] == True) or (self.subdivided[i] == 1): #set parent subtriangle properties to zero
                    self.p_mass[i] = 0
                    self.hpd_nodes[i] = 0
                    continue
                elif(self.subdivided[i] == False) or (self.subdivided[i] == 0):
                    sub_mass = toMpMath(self.gaussQuadSub(self.triangles[i]))
                    self.p_mass[i] = sub_mass
                    self.hpd_nodes[i] = 0 #nodes in HPD will come later
                else:
                    continue

                #accumulate the probability mass now
                self.total_mass += self.p_mass[i]
                if i % 5000 == 0:
                    print(f"{i}/{len(self.triangles)} subtriangle masses computed")

        elif(self.iterator == 0):
            for i in range(len(self.triangles)):
                sub_mass = toMpMath(self.gaussQuadSub(self.triangles[i]))
                self.p_mass.append(sub_mass)
                self.hpd_nodes.append(0) #will be calculated after boundary classification
                self.subdivided.append(False)

            #accumulate the probability mass now
                self.total_mass += self.p_mass[i]
                if i % 5000 == 0:
                    print(f"{i}/{len(self.triangles)} subtriangle masses computed")

        ### Uncomment to verify total domain mass ###
        #self.total_mass = total_mass
        self.p_mass = np.array(self.p_mass, dtype = object)
        print(f"total domain mass is {self.total_mass}")
        #print(self.p_mass)
        #print(self.subdivided)

    def calcHPDArea(self):


        self.integrateWithGQ() 
        t = self.getThresholdDensity(0.95)

        hpd_area = self.sumHPDTriangles()
        
        print(f"HPD Area for params {self.alphas} is {hpd_area}")

        if((self.alphas[0] < 0.9) or (self.alphas[1] < 0.9) or (self.alphas[2] < 0.9)): #or if total domain mass is < 0.99
            self.iterator = 1
            division_counts = 0
            while (self.total_mass < 0.99): #cap this to 10 for now, and also do tanh sinh vs subdivision logic for each step
                self.singularitySubDivide()
                self.integrateWithGQ()
                t = self.getThresholdDensity(0.95)
                hpd_area = self.sumHPDTriangles()
                            
                print(f"HPD Area for params {self.alphas} is {hpd_area} for iteration {division_counts + 2}")
                division_counts += 1

        print(f"the process took {division_counts} levels of subdivision")
        #self.iterator = 1


    #determine variance in near-edge PDF from centroid PDF for all three vertices.
    def varianceCalculate(self, subtriangle):
        on_border = False

        for i in range(3):
            if (self.vertices[subtriangle[i]][0] < 1e-15) or (self.vertices[subtriangle[i]][1] < 1e-15) or (self.vertices[subtriangle[i]][2] < 1e-15):
                on_border = True
            else:
                continue

        if on_border == True:
            centroid = self.localBCfromRefBC(subtriangle, [1/3,1/3,1/3])

            #calculate the three near-vertex points 
            nv1 = self.localBCfromRefBC(subtriangle, [0.95, 0.025, 0.025])
            nv2 = self.localBCfromRefBC(subtriangle, [0.025, 0.95, 0.025])
            nv3 = self.localBCfromRefBC(subtriangle, [0.025, 0.025, 0.95])

            #calculate pdfs of near-vertex pts and centroid
            pdfnodes = [toMpMath(dirichletpdf(nv1, self.alphas)), toMpMath(dirichletpdf(nv2, self.alphas)),\
                toMpMath(dirichletpdf(nv3, self.alphas)), toMpMath(dirichletpdf(centroid, self.alphas))]

            minpdf = toMpMath(min(pdfnodes))
            maxpdf = toMpMath(max(pdfnodes))

            max_var = maxpdf - minpdf

            if max_var > 1e1:
                ts_candidate = True
            elif max_var < 1e1:
                ts_candidate = False

            return ts_candidate
        else:
            return False

    #function to calculate area of HPD using info contained in self.hpd_nodes
    def sumHPDTriangles(self):
        hpd_area = 0
        for i in range(len(self.triangles)):
            if(self.hpd_nodes[i] == 13):
                hpd_area += self.getSubArea(self.triangles[i])
            elif(self.hpd_nodes[i] == 0):
                continue

        return hpd_area

    def singularitySubDivide(self): 
        divide_counts = 0

        self.length_record = len(self.triangles) #record number of subtriangles now
        self.total_mass = 0 #set total mass to zero for recalculation later
        #iterate through subtriangles, calculate variance, divide if variance exceeds threshold
        for i in range(len(self.triangles)):
            ts = self.varianceCalculate(self.triangles[i])
            if ((ts == True) and (self.subdivided[i] == 0)):
                self.quadTreeDivide(self.triangles[i])
                self.subdivided[i] = True
                self.p_mass[i] = 0
                self.hpd_nodes[i] = 0
                divide_counts +=1 #for counting number of triangles divided
            elif((ts == True) and (self.subdivided[i] == 1)):
                continue
            elif ts == False:
                self.total_mass += self.p_mass[i] #get partial p-mass for accumulation later via integrate with GQ function
                self.subdivided[i] = False
                continue
            

        #need to allocate more values to subtriangle property arrays
        p_mass_append = np.zeros(4 * divide_counts)
        self.p_mass = np.concatenate((self.p_mass, p_mass_append), dtype = object)
        self.hpd_nodes = np.concatenate((self.hpd_nodes, p_mass_append))
        self.subdivided = np.concatenate((self.subdivided, p_mass_append))


    #subdivide triangles with singular borders
    def quadTreeDivide(self, subtriangle):

        #calculate midpoints of subtriangle sides
        midpoint_1 = []
        midpoint_2 = []
        midpoint_3 = []

        #calculate bc coords of midpoints
        for i in range(3):
            lambda_01 = (self.vertices[subtriangle[0]][i] + self.vertices[subtriangle[1]][i]) / 2
            lambda_11 = (self.vertices[subtriangle[2]][i] + self.vertices[subtriangle[0]][i]) / 2
            lambda_21 = (self.vertices[subtriangle[1]][i] + self.vertices[subtriangle[2]][i]) / 2
            midpoint_1.append(lambda_01)
            midpoint_2.append(lambda_11)
            midpoint_3.append(lambda_21)

        '''
        self.vertices = self.vertices.tolist()
        self.index_vert = self.index_vert.tolist()
        '''
        
        #append these new points to our master vertices list
        self.vertices.append(midpoint_1)
        self.vertices.append(midpoint_2)
        self.vertices.append(midpoint_3)

        #add indices to our master vertex index list to be able to reference these new points
        index_1 = self.index_vert[-1][-1]+1 
        index_2 = self.index_vert[-1][-1]+2
        index_3 = self.index_vert[-1][-1]+3

        self.index_vert.append([index_1, index_2, index_3])

        self.triangles.append([subtriangle[0], index_1, index_2])
        self.triangles.append([index_1, index_2, index_3])
        self.triangles.append([index_1, subtriangle[1], index_3])
        self.triangles.append([index_2, index_3, subtriangle[2]])

        #self.hpd_nodes = self.hpd_nodes.tolist()
        #for i in range(4):
            #self.hpd_nodes.append(0)
            #self.p_mass.append(0)
        #self.hpd_nodes = np.array(self.hpd_nodes)

        '''
        self.vertices = np.array(self.vertices, dtype=object)
        self.index_vert = np.array(self.index_vert)

        self.index_vert = np.append(self.index_vert, [index_1, index_2, index_3], axis=0)
        '''
        
    #initial estimate for getting thresholding density+
    def getThresholdDensity(self, p):

        #sort subtriangles by density; not mass, because subtriangle area may vary
        tmp_density_array = []

        for i in range(len(self.p_mass)):
            tmp_density_array.append(self.p_mass[i] / self.getSubArea(self.triangles[i]))
        tmp_density_array = np.array([float(x) for x in tmp_density_array])
        p_mass_indices = np.argsort(tmp_density_array)[::-1] #get indices for largest -> smallest density triangles
        del tmp_density_array

        #get initial estimation for threshold density from a descending sort of density
        threshold_init = 0

        #self.hpd_nodes = np.zeros(len(self.p_mass))
        for i in range(len(self.p_mass)): #convert to cumulative sum
            if(threshold_init < p):
                #print("true")
                threshold_init += self.p_mass[p_mass_indices[i]] #accumulate probability mass from largest to smallest
                if threshold_init > p:
                    threshold_real = threshold_init - self.p_mass[p_mass_indices[i]] 
                self.hpd_nodes[p_mass_indices[i]] = 13 #temporarily classify subtriangle as in HPD 
                
            else:
                self.hpd_nodes[p_mass_indices[i]] = 0

        return threshold_init

        
    def meshRefinementTest(self, subtriangle):
        self.quadTreeDivide(subtriangle)


     #get bounds of integration for the singular subtriangle as a range [u0, u1]x[v0, v1]
    def genTanhSinhMeshUVBound(self, subtriangle):
        #alpha = [toMpMath(k) for k in alpha]
        #h = toMpMath(h)
        #v_ref = mp.mpf('0.2277858511416451') #one of the GQ nodes for a 20-pt rule
        v_ref = mp.mpf('0.9931285991850950')
        v_0 = toMpMath(0.5) * (v_ref + 1)
        #w_v = 0.5 * 0.1491729864726042
        #w_v = 0.5 * 0.0176140071391509
        k = 0
        t_ub = toMpMath(self.t_extrema)
        s_lb = toMpMath(self.s_min)
        s_ub = toMpMath(self.s_max)

        #calculates the summation term for truncation by magnitude for u or v, holding the other constant
        def calcSumTerm(t, uv_var):
            
            uv = mp.mpf('0.5') * mp.tanh((mp.pi / 2) * mp.sinh(t)) + mp.mpf('0.5')

            if(uv_var == 'u'):
                f = toMpMath(dirichletpdf(self.localBCfromUV(subtriangle, uv, v_0), self.alphas))
                u_jac = uv
            elif(uv_var == 'v'):
                #print(f"PROBLEM AREA-----{self.localBCfromUV(subtriangle, v_0, uv, orientation)}")
                f = toMpMath(dirichletpdf(self.localBCfromUV(subtriangle, v_0, uv), self.alphas))
                u_jac = v_0
            else:
                print("wrong u or v variable")
                quit()
            w_k = (mp.pi / 4) * (mp.cosh(t) / ((mp.cosh((mp.pi/2) * mp.sinh(t)))**2))
            #print(f"f is {f}, w_k is {w_k}, local bc is {self.localBCfromUV(subtriangle, u, v_0, 2)}")
            return f * w_k * self.res_ts**2 * (1 - u_jac) * 2 * self.getSubArea(subtriangle)#self.area * (toMpMath(1 / (self.res - 1)))**2# * w_v

        #for u to 1
        while True:
            t = t_ub + (k * self.res_ts)
            sum_term = calcSumTerm(t, 'u')
            #print(f"sum term for t is {sum_term}")

            if (sum_term < 1e-16) and (k > 0):
                #print(f"for t={t}, sumterm is {sum_term}")
                break
            elif (sum_term == inf) and (k > 0):
                #print(f"for t={t}, sumterm is {sum_term}")
                t -= self.res_ts
                #print(f"for new t={t} with h={h}...sum term is {calcSumTerm(t, 'u', orientation)}")
                break
            elif (mp.isnan(sum_term) == True):
                #print(f"for t={t}, sumterm is {sum_term}")
                t -= self.res_ts
                #print(f"for new t={t} with h={h}...sum term is {calcSumTerm(t, 'u', orientation)}")
                break
            #elif (sum_term == inf) and (k == 0):
            #    print("error: t bound started at inf")
            #    quit()
            else:
                k += 1

        #for v to 0
        k=0
        while True:
            s_0 = s_lb - (k * self.res_ts)
            sum_term = calcSumTerm(s_0, 'v')

            if (sum_term < 1e-40) and (k > 0):
                break
            elif(sum_term == inf) and (k > 0):
                #print(f"for s_0={s_0}, sumterm is {sum_term}")
                s_0 += self.res_ts
                #print(f"for new s_0={s_0}, sumertm is {calcSumTerm(s_0, 'v', orientation)}")
                break
            elif (mp.isnan(sum_term) == True):
                #print(f"for s0={s_0}, sumterm is {sum_term}")
                s_0 -= self.res_ts
                #print(f"for new s0={s_0} with h={h}...sum term is {calcSumTerm(s_0, 'u', orientation)}")
                break
            #elif (sum_term == inf) and (k == 0):
             #   print("error: t bound started at inf")
            #    quit()
            else:
                k +=1

        #for v to 1
        k=0
        while True:
            s_1 = s_ub + (k * self.res_ts)
            sum_term = calcSumTerm(s_1, 'v')

            if (sum_term < 1e-40) and (k > 0):
                #print(f"for t={t}, sumterm is {sum_term}")
                break
            elif (sum_term == inf) and (k > 0):
                #print(f"for s_1={s_1}, sumterm is {sum_term}")
                s_1 -= self.res_ts
                #print(f"for new s_1={s_1} with h={h}...sum term is {calcSumTerm(s_1, 'v', orientation)}")
                break
            elif (mp.isnan(sum_term) == True):
                #print(f"for s1={s_1}, sumterm is {sum_term}")
                s_1 -= self.res_ts
                #print(f"for new s1={s_1} with h={h}...sum term is {calcSumTerm(s_1, 'u', orientation)}")
                break
            #elif (sum_term == inf) and (k == 0):
            #    print("error: t bound started at inf")
            #    quit()
            else:
                k += 1

        return [[-t, t], [s_0, s_1]]

     #function for integrating the singular subtriangle using tanh-sinh quadrature
    def tanhSinhIntegral(self, subtriangle):

        #define bounds
        st_bound = self.genTanhSinhMeshUVBound(subtriangle)
        t_bound = [toMpMath(a) for a in st_bound[0]]
        s_bound = [toMpMath(b) for b in st_bound[1]]
        t_ub_refined = toMpMath(t_bound[1] - mp.mpf('0.1'))#
        s_ub_refined = toMpMath(s_bound[1] - mp.mpf('0.1'))#
        s_lb_refined = toMpMath(s_bound[0] + mp.mpf('0.1'))
        h_1 = mp.mpf('0.1') #for the middle intervals [0+epsilon, 1-epsilon]

        #print(f"The domain interval for method 2 is [{t_bound[0], t_bound[1]}] x [{s_bound[0], s_bound[1]}]")

        #define interval step sizes
        t_step = int((t_bound[1]-t_ub_refined)/self.res_ts)
        s_step = int((s_bound[1]-s_ub_refined)/self.res_ts)
        s_coarse = int((s_ub_refined - s_lb_refined)/h_1)
        t_coarse = int((t_ub_refined - t_bound[0])/h_1)
###### ATTEMPT AT INCORPORATING SUBSUBTRIANGLE PROPERTIES INTO SUBTRIANGLE LISTS
        #define interval of coarse integration via step size h_1
        k_coarse = mp.linspace(mp.mpf(t_bound[0]), t_ub_refined, t_coarse, endpoint=False)
        l_coarse = mp.linspace(s_lb_refined, s_ub_refined, s_coarse, endpoint=False)

        #define intervals of fine integration via h
        k_refined = mp.arange(t_ub_refined, t_bound[1], self.res_ts)
        l_refined_lower = mp.arange(mp.mpf(s_bound[0]), s_lb_refined, self.res_ts)
        l_refined_upper = mp.arange(s_ub_refined, mp.mpf(s_bound[1]), self.res_ts)
        k_refined.append(t_bound[1])
        l_refined_upper.append(mp.mpf(s_bound[1]))

        #concatenate intervals to get domain of tanh-sinh function
        k = k_coarse + k_refined
        l = l_refined_lower + l_coarse + l_refined_upper

        #range of tanh sinh function calculated below; domain of integration of Dirichlet PDF
        v_l = [0.5 * mp.tanh((mp.pi / 2) * mp.sinh(t)) + 0.5 for t in l]
        u_k = [0.5 * mp.tanh((mp.pi / 2) * mp.sinh(t)) + 0.5 for t in k]

        summation = 0

        #for every point (u_i, v_j), calculate the PDF, weight, jacobian, and area given by tanh-sinh
        for i in range(len(v_l)):
            for j in range(len(u_k)):
                g_t = (mp.pi / 4) * (mp.cosh(k[j]) / ((mp.cosh((mp.pi/2) * mp.sinh(k[j])))**2))
                g_s = (mp.pi / 4) * (mp.cosh(l[i]) / ((mp.cosh((mp.pi/2) * mp.sinh(l[i])))**2))
                weights = g_t * g_s
                jacobian = toMpMath(1 - u_k[j])

                if((i==0) or (i==(len(v_l)-1))):
                    h_v = self.res_ts
                else:
                    h_v = (l[i+1] - l[i-1])/2

                if(j==0):
                    h_u = h_1
                elif(j==(len(u_k)-1)):
                    h_u = self.res_ts
                else:
                    h_u = (k[j+1]-k[j-1])/2

                value = dirichletpdf(self.localBCfromUV(subtriangle, u_k[j], v_l[i]), self.alphas) * weights * jacobian * h_u * h_v
                summation += value
            if(i%100==0):
                print(f"{i}/{len(v_l)-1} quadrature columns computed")
        pdf_integral = summation * 2 * self.getSubArea(subtriangle)#area * (1 / (self.res - 1))**2
        self.mass += pdf_integral
        print(f"Tanh-sinh yields {pdf_integral} for subtriangle {subtriangle}")
        plt.figure(1)
        x_list = []
        y_list = []
            
        for i in range(len(subtriangle)):
            x, y = bc2xy(self.vertices[subtriangle[i]])
            x_list.append(x)
            y_list.append(y)
        plt.plot(x_list, y_list, 'r.')
        return pdf_integral

    def testTSBounds(self):
        return(self.tanhSinhIntegral(self.triangles[0]))


    #function for calculating subtriangle barycentric coordinates from U, V coordinates
    #add orientation functionality, which exists in experiment branch
    def localBCfromUV(self, subtriangle, u, v):
        u = toMpMath(u)
        v = toMpMath(v)
        lambdas = uv2bc(u, v)

        coords = self.localBCfromRefBC(subtriangle, lambdas)
        return coords

    




if __name__ == "__main__":
    main()