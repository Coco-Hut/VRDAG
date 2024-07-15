import numpy as np
import networkx as nx
import powerlaw
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import connected_components
import warnings
warnings.filterwarnings('ignore')

def wedge_count(A_in):
    """
    Compute the wedge count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    The wedge count.
    """
    degrees = A_in.sum(axis=0)
    return float(np.sum(np.array([0.5 * x * (x - 1) for x in degrees])))

def claw_count(A_in):
    """
    Compute the claw count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Claw count
    """

    degrees = A_in.sum(axis=0)
    return float(np.sum(np.array([1 / 6. * x * (x - 1) * (x - 2) for x in degrees])))

def triangle_count(A_in):
    """
    Compute the triangle count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Triangle count
    """

    A_graph = nx.from_numpy_matrix(A_in)
    triangles = nx.triangles(A_graph)
    t = np.sum(list(triangles.values())) / 3
    return int(t)

def gini(A_in,flow='out'):
    """
    Compute the Gini coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Gini coefficient
    """

    n = A_in.shape[0]
    if flow=='out':
        degrees = A_in.sum(axis=0)
    else:
        degrees = A_in.sum(axis=1)
    degrees_sorted = np.sort(degrees)
    G = (2 * np.sum(np.array([i * degrees_sorted[i] for i in range(len(degrees))]))) / (n * np.sum(degrees)) - (
            n + 1) / n
    return float(G)

def LCC_size(A_in):
    """
    Compute the size of the largest connected component (LCC)

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Size of LCC

    """
    unique, counts = np.unique(connected_components(A_in)[1], return_counts=True)
    LCC = len(np.where(connected_components(A_in)[1] == np.argmax(counts))[0])

    return LCC

def power_law_exp(A_in,flow='out'):
    
    if flow=='out':
        degrees=A_in.sum(axis=0)
    elif flow=='in':
        degrees=A_in.sum(axis=1)
    else:
        raise ValueError('This flow direction does not exist!')
    
    return powerlaw.Fit(degrees, xmin=max(np.min(degrees), 1)).power_law.alpha

def node_div_dist(A_in):
    
    out_degree=A_in.sum(axis=0) 
    in_degree=A_in.sum(axis=1) 
    
    max_degree=np.where(out_degree>in_degree,out_degree,in_degree)
    max_degree=np.where(max_degree>0,max_degree,1)
    
    return ((out_degree-in_degree)/max_degree).reshape(-1,1) # [N,1]

def calculate_mmd(x1, x2, beta):
    x1x1 = gaussian_kernel(x1, x1, beta)
    x1x2 = gaussian_kernel(x1, x2, beta)
    x2x2 = gaussian_kernel(x2, x2, beta)
    diff = x1x1.mean() - 2 * x1x2.mean() + x2x2.mean()

    #print("MMD means", x1x1.mean(),x1x2.mean(),x2x2.mean())

    return diff

def gaussian_kernel(x1, x2, beta = 1.0):
    L=pairwise_distances(x1,x2).reshape(-1)
    return np.exp(-beta*np.square(L))