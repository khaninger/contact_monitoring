import pickle
import numpy as np
from sklearn import mixture
from segmentation import segmentation


f = pickle.load(open('data/video_sexy_rake_hinge_3.pickle', 'rb'))
f = np.array(f)
print(f)
f = f[:,:,:4]
print(f.shape)
key_points = f[:,:,:3]
key1 = np.zeros((f.shape[0],key_points.shape[2]))
key2 = np.zeros((f.shape[0],key_points.shape[2]))
key3 = np.zeros((f.shape[0],key_points.shape[2]))
key4 = np.zeros((f.shape[0],key_points.shape[2]))
time = []
for i in range(f.shape[0]):
    time.append(f[i][0,3])
time = np.array(time).reshape(len(time),)
for ii in range(key_points.shape[0]):
    for jj in range(key_points.shape[2]):
        key1[ii,jj] = key_points[ii,0,jj]
        key2[ii,jj] = key_points[ii,1,jj]
        key2[ii,jj] = key_points[ii,2,jj]
        key2[ii,jj] = key_points[ii,3,jj]

traj = np.column_stack((time,key1,key2,key3,key4))
np.random.seed(42)
n_cl = 3 # number of clusters
gmm = mixture.GaussianMixture(n_components=n_cl, covariance_type='full', reg_covar=10 ** -6).fit(traj)
idx = segmentation(gmm,time,n_cl)
print(idx)

f_new = np.zeros((f.shape[0],f.shape[1], f.shape[2]+1))
print(f_new.shape)
for kk in range(f.shape[0]):
    if kk<idx[0]:
        f_new[kk] = np.column_stack((f[kk],np.zeros(f.shape[1])))
    elif kk<idx[1]:
        f_new[kk] = np.column_stack((f[kk],np.ones(f.shape[1])))
    else:
        f_new[kk] = np.column_stack((f[kk],np.ones(f.shape[2])))
print(f_new[0])

#filename = 'clustering_video_sexy_rake_hinge_3.pickle'
#outfile = open(filename,'wb')
#pickle.dump(f_new,outfile)
#outfile.close()