import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_points(L,rest_pt):

    if L[0].shape == (4,4):
        points = np.array([arr[:3, 3] for arr in L])
    else:
        points = (np.array(L))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b')
    ax.scatter(rest_pt[0], rest_pt[1], rest_pt[2], color='r')
    plt.show()


def plot_3d_points_segments(L, rest_pt, exp_n=0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color_segments = ['b', 'g', 'm']
    for i, L_segment in enumerate(L):
        if L_segment[0].shape == (4,4):
            points = np.array([arr[:3, 3] for arr in L])
        else:
            points = (np.array(L_segment))
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color_segments[i], label="segment "+str(i))
    if np.mean(rest_pt) != 0:
        ax.scatter(rest_pt[0], rest_pt[1], rest_pt[2], color='r', label="rest_pt")
    if exp_n != 0:
        ax.set_title(f"Experiment {exp_n}")
    ax.legend()
    plt.show()

def plot_distance(L,rest_pt):
    if L[0].shape == (4, 4):
        distances = np.array([np.linalg.norm(arr[:3, 3]-rest_pt) for arr in L])
    else:
        distances = np.array([np.linalg.norm(arr - rest_pt) for arr in L])
    print(f"distances: {distances}")


if __name__ == "__main__":
    from dataload_helper import plug, rake
    show_plug = False
    show_rake = True
    if show_plug:

        for i in range(3):
            dataset = plug(index=1 + i, experiment="", clustered=False, segment=True).load()
            plot_3d_points_segments(dataset, [-.3, -.3, 0])

        for i in range(3):
            dataset = plug(index=1 + i, experiment="less_", clustered=False, segment=True).load()
            plot_3d_points_segments(dataset, [-.3, -.3, 0])

        for i in range(3):
            dataset = plug(index=1 + i, experiment="threading_", clustered=False, segment=True).load()
            plot_3d_points_segments(dataset, [-.3, -.3, 0])
    if show_rake:
        for i in range(5):
            dataset = rake(index=i+1, clustered=False, segment=True).load(center=True)
            plot_3d_points_segments(dataset, [0,0,0], i+1)