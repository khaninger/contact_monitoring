import numpy as np
from scipy.stats import norm, multivariate_normal
import math


def segmentation(gmm,time,n_cl, return_lik=False):

    means_time = gmm.means_[:, 0]
    overall_cov_tensor = gmm.covariances_
    overall_weights = gmm.weights_
    time_variance = overall_cov_tensor[:,0,0]
    total = np.sum(overall_weights)
    likelihood = np.zeros( (len(time),n_cl))
    for i in range(n_cl):
        distribution = norm(loc=means_time[i], scale=math.sqrt(time_variance[i]))
        likelihood[:, i] = distribution.pdf(time)

    numerator = likelihood * overall_weights
    denominator = numerator.sum(axis=1)
    time_weights = np.zeros((len(time), n_cl))
    for i in range(n_cl):
        time_weights[:,i] = numerator[:,i]/denominator
    idx = np.zeros(n_cl-1)
    for i in range(n_cl-1):
        idx[i] = np.argwhere(np.diff(np.sign(time_weights[:,i] - time_weights[:,i+1])) != 0).flatten()

    idx = idx.astype(int)
    if not return_lik:
        return idx
    else:
        return idx, likelihood
