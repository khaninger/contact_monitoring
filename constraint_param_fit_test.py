from dataload_helper import plug
from visualize import plot_3d_points, plot_distance, plot_3d_points_segments
import numpy as np
from constraint import PointConstraint, CableConstraint

dataset = np.array(plug(index=3, experiment="threading_", clustered=False, segment=True).load())
