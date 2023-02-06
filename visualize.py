import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_points(L,rest_pt):
    points = np.array([arr[:3, 3] for arr in L])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.scatter(rest_pt[0], rest_pt[1], rest_pt[2], color='r')
    plt.show()

def plot_distance(L,rest_pt):
    distances = np.array([np.linalg.norm(arr[:3, 3]-rest_pt) for arr in L])
    print(f"distances: {distances}")