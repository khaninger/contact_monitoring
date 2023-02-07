from dataload_helper import plug
from visualize import plot_3d_points, plot_distance, plot_3d_points_segments
import numpy as np
from constraint import PointConstraint, CableConstraint

dataset = np.array(plug(index=3, experiment="threading_", clustered=False, segment=True).load())


segment_2 = dataset[2]
constraint = CableConstraint()
params = constraint.fit(segment_2)
plot_3d_points_segments(dataset, params['rest_pt'])
#plot_3d_points_segments(dataset, [0,0,0])
#plot_distance(segment_2, params['rest_pt'])