import numpy as np
import matplotlib.pyplot as plt
from segmentation import segmentation
import pickle
from sklearn import mixture


np.random.seed(42)
n_cl = 2  # number of clusters
f = pickle.load(open('video_plug_3.p', 'rb'))
f = np.array(f)
time = f[:,3]
multi_traj = f[:,0:3]
traj = np.column_stack((time,multi_traj))
dof = multi_traj.shape[1]
gmm = mixture.GaussianMixture(n_components=n_cl, covariance_type='full', reg_covar=10 ** -6).fit(traj)
idx = segmentation(gmm)
print(idx)
x = multi_traj[:,0]
y = multi_traj[:,1]
z = multi_traj[:,2]
# creating labels series
labelling = np.zeros(len(time))
for i in range(len(time)):
    if i<idx[0]:
        labelling[i] = 0
    else:
        labelling[i] = 1




print(labelling)
data_video_plug_3 = np.column_stack((x, y, z, time, labelling))
#filename = 'clustering_video_plug_3.pickle'
#outfile = open(filename,'wb')
#pickle.dump(data_video_plug_3,outfile)
#outfile.close()
