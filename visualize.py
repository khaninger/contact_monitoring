import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_3d_points(L,rest_pt):
    color_segments = ['b', 'g', 'm']
    if L[0].shape == (4,4):
        points = np.array([arr[:3, 3] for arr in L])
    else:
        points = (np.array(L))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(points.shape[1]):
        ax.scatter(points[:, i, 0], points[:, i, 1], points[:, i, 2], color=color_segments[i])
    ax.scatter(rest_pt[0], rest_pt[1], rest_pt[2], color='r')
    set_axes_equal(ax)
    plt.show()


def plot_3d_points_segments(L, rest_pt=np.array([-0.31187662, -0.36479221, -0.03707742]), radius = .28, exp_n=0):
    fig = plt.figure(figsize=(14, 10), dpi=80)
    # Create a sphere using spherical coordinates
    phi = np.linspace(0, np.pi, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    x = radius*np.outer(np.sin(phi), np.cos(theta)) + rest_pt[0]
    y = radius*np.outer(np.sin(phi), np.sin(theta)) + rest_pt[1]
    z = radius*np.outer(np.cos(phi), np.ones_like(theta)) + rest_pt[2]

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z, color='black', linewidth=0.5)
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
    set_axes_equal(ax)
    ax.legend()
    plt.show()

def plot_x_pt_inX(L_pt, X):
    colors = ['b', 'g', 'm']
    labels = ['pt_0', 'pt_1', 'pt_2']

    fig = plt.figure(figsize=(14, 10), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    #set_axes_equal(ax)
    for T in X:
        for i, pt in enumerate(L_pt):
            point = (T @ np.append(pt,0).T)[:3]
            ax.scatter(point[0], point[1], point[2], color=colors[i])
    ax.set_title(f"pt in world coordinates")
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
    show_rake = False
    if show_plug:

        for i in range(3):
            dataset = plug(index=1 + i, experiment="", clustered=True, segment=True).load()
            plot_3d_points_segments(dataset, [-.3, -.3, 0])
        """
        for i in range(3):
            dataset = plug(index=1 + i, experiment="less_", clustered=False, segment=True).load()
            plot_3d_points_segments(dataset, [-.3, -.3, 0])

        for i in range(3):
            dataset = plug(index=1 + i, experiment="threading_", clustered=False, segment=True).load()
            plot_3d_points_segments(dataset, [-.3, -.3, 0])
        """
    if show_rake:
        for i in range(5):
            dataset = rake(index=i+1, clustered=False, segment=True).load(center=True)
            plot_3d_points_segments(dataset, [0,0,0], i+1)


if __name__ == "__main__":
    from dataload_helper import point_c_data

    pts = point_c_data(n_points=3, noise=0.01).points()
    plot_3d_points(L=pts, rest_pt = np.array([0.1, 0.2, 0.3]))