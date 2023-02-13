import numpy as np




def find_T(A, B):

    assert A.shape == B.shape
    # get index of comparable keypoints
    KP_bool = []
    KP_0_bool = np.sum(A, axis=1) != 0
    KP_1_bool = np.sum(B, axis=1) != 0
    for i in range(KP_0_bool.shape[0]):
        KP_bool.append(KP_1_bool[i] and KP_0_bool[i])
    # omit keypoint that cant be compared (not detected keypoints)
    A = A[KP_bool]
    B = B[KP_bool]

    A = A.T
    B = B.T

    dim = A.shape[0]
    if dim != 3:
        A = A.T

    dim = B.shape[0]
    if dim != 3:
        B = B.T
        dim = B.shape[0]

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    AA = A - centroid_A
    BB = B - centroid_B

    # Calculate cov matrix
    H = AA @ np.transpose(BB)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        #print("det(R) < 0, correcting reflection")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    # homogeneous transformation
    T_A2B = np.identity(dim + 1)
    T_A2B[:dim, :dim] = R
    T_A2B[:dim, dim] = t.flatten()

    return T_A2B

def init_T(keypoint_array):
    """
    Generates Object pose from Keypoints
    center of pose is mean of all keypoints that are not zero but first keypoint aka center keypoint -> [1:]
    """
    t_object = np.mean(keypoint_array, axis=0)
    T_object = np.eye(4)
    T_object[:3, 3] = t_object
    return T_object