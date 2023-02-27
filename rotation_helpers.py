import casadi as ca
import numpy as np

def cross(a,b):
    return ca.vertcat(a[1]*b[2]-a[2]*b[1],
                      a[2]*b[0]-a[0]*b[2],
                      a[0]*b[1]-a[1]*b[0])

def dist_T(T1, T2):
    return np.linalg.norm(T1[:3,3]-T2[:3,3])

def transform_pt(T, x):
    if type(T) is np.ndarray:
        T = ca.DM(T)
    return (T @ ca.vertcat(x, ca.SX(1)))[:3]

def invert_TransMat(T_a2b):
    T_b2a = np.eye(4)
    R_b2a = T_a2b[0:3, 0:3].T
    t_b2a = -R_b2a @ T_a2b[0:3, 3]
    T_b2a[0:3, 0:3] = R_b2a
    T_b2a[0:3, 3] = t_b2a
    return T_b2a

def transform_force(T, f):
    # IN: T should be the transformation matrix for the pose of TCP frame in base frame
    # IN: f is force in TCP frame
    pos = T[:3,-1]
    R = T[:3, :3]
    cross_product = np.array([[0,      -pos[-1], pos[1]],
                              [pos[-1], 0,      -pos[0]],
                              [-pos[1], pos[0],  0]])
    force_transform  = np.hstack([np.vstack([R,cross_product@R]),
                                  np.vstack([np.zeros((3,3)), R])])
    return force_transform@f # returns the force in base frame

#### Rotation vectors ####
def rotvec_to_rotation(vec):
    ty = ca.SX if type(vec) is ca.SX else ca.DM
    rot = ty.zeros(3,3)
    phi_ = ca.sqrt(ca.sumsqr(vec))
    phi = ca.if_else(phi_<1e-9, 1e-9, phi_)
    #phi = ca.if_else(phi>2*np.pi, phi-2*np.pi, phi)
    kx = vec[0]/phi
    ky = vec[1]/phi
    kz = vec[2]/phi

    cp = ca.cos(phi)
    sp = ca.sin(phi)
    vp = 1-cp
    rot[0,0] =  kx*kx*vp   +cp
    rot[0,1] =  kx*ky*vp-kz*sp
    rot[0,2] =  kx*kz*vp+ky*sp
    rot[1,0] =  kx*ky*vp+kz*sp
    rot[1,1] =  ky*ky*vp   +cp
    rot[1,2] =  ky*kz*vp-kx*sp
    rot[2,0] =  kx*kz*vp-ky*sp
    rot[2,1] =  ky*kz*vp+kx*sp
    rot[2,2] =  kz*kz*vp   +cp
    return rot

def rotation_to_quat(R):
    r = rotation_to_rotvec(R)
    return rotvec_to_quat(r)

def rotation_to_rotvec(R):
    # http://scipp.ucsc.edu/~haber/ph216/rotation_12.pdf, (20), (27), (32)
    # also see scipy https://github.com/scipy/scipy/blob/a4bed793057026b86dc8fb12a5fd69813da8a728/scipy/spatial/transform/_rotation.pyx#L879

    tr = ca.trace(R)
    theta = ca.acos(0.5*(tr-1)) # rotation about the axis in radians

    r = ca.vertcat(R[2,1]-R[1,2], # axis about which the rotation occurs
                   R[0,2]-R[2,0],
                   R[1,0]-R[0,1])

    # There's two degenerate cases, when tr = -1 or 3
    # We handle that in two nested if_else's,
    # normalizing r only when both are false. 
    
    r = ca.if_else(ca.fabs((tr + 1.0))<1e-9,
                   ca.vertcat(ca.sqrt(0.5*(1+R[0,0])),
                              ca.sqrt(0.5*(1+R[1,1])),
                              ca.sqrt(0.5*(1+R[2,2]))),
                   ca.if_else(ca.fabs((tr - 3.0))<1e-9,
                              ca.DM.zeros(3),
                              r/ca.sqrt((3-tr)*(1+tr))))

    return r*theta

def tmat_to_rotvec_pose(T):
    pos = T[:3,-1]
    rot = T[:3,:3]
    rot_vec = rotation_to_rotvec(rot)
    return ca.vertcat(pos,rot_vec)

def rotvec_pose_to_tmat(r):
    R_sym = rotvec_to_rotation(r[3:])
    rot = ca.vertcat(R_sym, ca.SX(1,3))
    pos = ca.vertcat(r[:3], ca.SX(1))
    return ca.horzcat(rot,pos)  # simbolic transformation matrix

def rotvec_to_quat(r):
    norm_r = ca.norm_2(r)
    th_2 = norm_r/2.0
    return ca.vertcat(ca.cos(th_2),
                      ca.sin(th_2)*r[0]/norm_r,
                      ca.sin(th_2)*r[1]/norm_r,
                      ca.sin(th_2)*r[2]/norm_r)

def rotvec_rotvec_mult(r1, r2):
    a1 = ca.norm_2(r1)
    a2 = ca.norm_2(r2)
    diff = (a2-a1)/2.0
    add  = (a2+a1)/2.0
    r1 *= 1/a1
    r2 *= 1/a2

    a = 2*ca.acos((1-r1.T@r2)*ca.cos(diff)-(1+r1.T@r2)*ca.cos(add))
    r = (ca.sin(add)+ca.sin(diff))*r1+(ca.sin(add)-ca.sin(diff))*r2+(ca.cos(diff)-ca.cos(add))*cross(r2,r1)
    #a = 2*ca.acos(ca.cos(a2/2)*ca.cos(a1/2)-ca.sin(a2/2)*ca.sin(a1/2)*(r1.T@r2))
    #r = ca.sin(a2/2)*ca.cos(a1/2)*r2+ca.sin(a1/2)*ca.cos(a2/2)*r1+ca.sin(a2/2)*ca.sin(a1/2)*cross(r2,r1)

    return a*r/(ca.sin(a/2))

def rotvec_vec_mult(r,v):
    a = ca.norm_2(r)
    r *= 1/a
    return ca.cos(a)*v+ca.sin(a)*cross(r,v)+(1-ca.cos(a))*(r.T@v)*r

#### ZYZ Euler angles ####
# Convert to quaternion from intrinsic ZYZ euler angles
# https://ntrs.nasa.gov/api/citations/19770024290/downloads/19770024290.pdf
def euler_to_quat(eu):
    return ca.vertcat(ca.cos(eu[1]/2)*ca.cos(eu[2]/2+eu[0]/2),
                      ca.sin(eu[1]/2)*ca.sin(eu[2]/2-eu[0]/2),
                      ca.sin(eu[1]/2)*ca.cos(eu[2]/2-eu[0]/2),
                      ca.cos(eu[1]/2)*ca.sin(eu[2]/2+eu[0]/2))

# Transform ZYZ Euler angles into a rotation matrix
def euler_to_rotation(eu):
    rot = ca.SX.zeros(3,3)
    rot[0,0] =  ca.cos(eu[0])*ca.cos(eu[1])*ca.cos(eu[2]) - ca.sin(eu[0])*ca.sin(eu[2])
    rot[0,1] = -ca.cos(eu[0])*ca.cos(eu[1])*ca.sin(eu[2]) - ca.sin(eu[0])*ca.cos(eu[2])
    rot[0,2] =  ca.cos(eu[0])*ca.sin(eu[1])
    rot[1,0] =  ca.sin(eu[0])*ca.cos(eu[1])*ca.cos(eu[2]) + ca.cos(eu[0])*ca.sin(eu[2])
    rot[1,1] = -ca.sin(eu[0])*ca.cos(eu[1])*ca.sin(eu[2]) + ca.cos(eu[0])*ca.cos(eu[2])
    rot[1,2] =  ca.sin(eu[0])*ca.sin(eu[1])
    rot[2,0] = -ca.sin(eu[1])*ca.cos(eu[2])
    rot[2,1] =  ca.sin(eu[1])*ca.sin(eu[2])
    rot[2,2] =  ca.cos(eu[1])
    return rot

# Convert a rotation matrix to ZYZ euler coordinates
def rotation_to_euler(rot):
    eu = ca.SX.zeros(3)
    epsilon = 0.999991
    eu[1] = ca.if_else(rot[2,2]>epsilon,
                       0,
                       ca.if_else(rot[2,2]<-epsilon, np.pi, ca.acos(rot[2,2])))
    eu[0] = ca.if_else(ca.fabs(rot[2,2]) >= epsilon,
                       0,
                       ca.atan2(rot[1,2], rot[0,2]))
    eu[2] = ca.if_else(ca.fabs(rot[2,2]) >= epsilon,
                       ca.atan2(rot[1,0], rot[1,1]),
                       ca.atan2(rot[2,1], -rot[2,0]))
    return eu

def eulerpose_to_rotpose(eu):
    ret = np.zeros(6)
    ret[:3] = eu[:3]
    ret[3:] = np.squeeze(quat_to_rotvec(euler_to_quat(eu[3:])))
    return ret

def eulerpose_to_quatpose(x):
    ret = np.zeros(7)
    ret[:3] = x[:3]
    ret[3:] = np.squeeze(euler_to_quat(x[3:]))
    return ret

#### XYZ Euler angles ####
# Convert to quaternion from extrinsic xyz euler angles
def xyz_to_quat(xyz): # Possible to optimize?
    ty = ca.SX if type(xyz) is ca.SX else ca.DM
    q0 = ca.horzcat(ca.cos(xyz[0]/2), ca.sin(xyz[0]/2), ty(0.0), ty(0.0))
    q1 = ca.horzcat(ca.cos(xyz[1]/2), ty(0.0), ca.sin(xyz[1]/2), ty(0.0))
    q2 = ca.horzcat(ca.cos(xyz[2]/2), ty(0.0), ty(0.0), ca.sin(xyz[2]/2))
    # extrinsic = intrinsic with reversed order
    return quat_quat_mult(q0,quat_quat_mult(q1,q2))

def xyz_to_rotation(eu):
    #return quat_to_rotation(xyz_to_quat(xyz)) # old method
    rot = ca.SX.zeros(3,3)
    rot[0,0] =  ca.cos(eu[1])*ca.cos(eu[2])
    rot[0,1] = -ca.cos(eu[1])*ca.sin(eu[2])
    rot[0,2] =  ca.sin(eu[1])
    rot[1,0] =  ca.sin(eu[0])*ca.sin(eu[1])*ca.cos(eu[2]) + ca.cos(eu[0])*ca.sin(eu[2])
    rot[1,1] = -ca.sin(eu[0])*ca.sin(eu[1])*ca.sin(eu[2]) + ca.cos(eu[0])*ca.cos(eu[2])
    rot[1,2] = -ca.sin(eu[0])*ca.cos(eu[1])
    rot[2,0] = -ca.cos(eu[0])*ca.sin(eu[1])*ca.cos(eu[2]) + ca.sin(eu[0])*ca.sin(eu[2])
    rot[2,1] =  ca.cos(eu[0])*ca.sin(eu[1])*ca.sin(eu[2]) + ca.sin(eu[0])*ca.cos(eu[2])
    rot[2,2] =  ca.cos(eu[0])*ca.cos(eu[1])
    return rot

def rotation_to_xyz(R):
    xyz = ca.SX.zeros(3)
    xyz[0] = ca.atan2(-R[1,2], R[2,2])
    xyz[1] = ca.atan2(R[0,2], ca.sqrt(1-R[0,2]*R[0,2]))
    xyz[2] = ca.atan2(-R[0,1], R[0,0])
    return xyz

def xyz_pose_to_tmat(x):
    R_sym = xyz_to_rotation(x[3:])
    rot = ca.vertcat(R_sym, ca.SX(1,3))
    pos = ca.vertcat(x[:3], ca.SX(1))
    return ca.horzcat(rot,pos)  # simbolic transformation matrix

def tmat_to_xyz_pose(T):
    pos = T[:3,-1]
    rot = T[:3,:3]
    rot_vec = rotation_to_xyz(rot)
    return ca.vertcat(pos,rot_vec)


#### Quaternions ####
#### Mostly following tutorial from Weizmann
def quat_to_rotvec(q):
    q *= ca.sign(q[0])  # multiplying all quat elements by negative 1 keeps same rotation, but only q0 > 0 works here
    th_2 = ca.acos(q[0])
    th = th_2*2.0
    rotvec = ca.vertcat(q[1]/ca.sin(th_2)*th, q[2]/ca.sin(th_2)*th, q[3]/ca.sin(th_2)*th)
    return rotvec

def quat_vec_mult(q,v):
    if type(q) is ca.SX:
        ret = ca.SX.zeros(3)
    elif type(q) is ca.MX:
        ret = ca.MX.zeros(3)
    else:
        ret = ca.DM.zeros(3)
    ret[0] =    v[0]*(q[0]**2+q[1]**2-q[2]**2-q[3]**2)\
             +2*v[1]*(q[1]*q[2]-q[0]*q[3])\
             +2*v[2]*(q[0]*q[2]+q[1]*q[3])
    ret[1] =  2*v[0]*(q[0]*q[3]+q[1]*q[2])\
             +  v[1]*(q[0]**2-q[1]**2+q[2]**2-q[3]**2)\
             +2*v[2]*(q[2]*q[3]-q[0]*q[1])
    ret[2] =  2*v[0]*(q[1]*q[3]-q[0]*q[2])\
             +2*v[1]*(q[0]*q[1]+q[2]*q[3])\
             +  v[2]*(q[0]**2-q[1]**2-q[2]**2+q[3]**2)
    return ret

def quat_quat_mult(q,p):
    if type(q) is ca.SX:
        ret = ca.SX.zeros(4)
    else:
        ret = ca.DM.zeros(4)
    ret[0] = q[0]*p[0]-q[1]*p[1]-q[2]*p[2]-q[3]*p[3]
    ret[1] = q[0]*p[1]+q[1]*p[0]-q[2]*p[3]+q[3]*p[2]
    ret[2] = q[0]*p[2]+q[1]*p[3]+q[2]*p[0]-q[3]*p[1]
    ret[3] = q[0]*p[3]-q[1]*p[2]+q[2]*p[1]+q[3]*p[0]
    return ret

def quaternion_inv(q):
    r = deepcopy(q)
    r[1:] *= -1.0
    return r

def quat_to_rotation(q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    if type(q) is ca.SX:
        r = ca.SX.zeros(3,3)
    else:
        r = ca.DM.zeros(3,3)
    r[0,0] = 2 * (q0 * q0 + q1 * q1) - 1
    r[0,1] = 2 * (q1 * q2 - q0 * q3)
    r[0,2] = 2 * (q1 * q3 + q0 * q2)
     
    r[1,0] = 2 * (q1 * q2 + q0 * q3)
    r[1,1] = 2 * (q0 * q0 + q2 * q2) - 1
    r[1,2] = 2 * (q2 * q3 - q0 * q1)
     
    r[2,0] = 2 * (q1 * q3 - q0 * q2)
    r[2,1] = 2 * (q2 * q3 + q0 * q1)
    r[2,2] = 2 * (q0 * q0 + q3 * q3) - 1
    return r

#### CONVENIENCE FOR TRANSFORMATIONS ####
# Transform from a relative pose in compliance frame x_c to world, based on init_pose
def comp_to_world(init_pose, x_c):
    q0 = rotvec_to_quat(init_pose[3:])
    x_w = quat_vec_mult(q0, x_c[:3])+init_pose[:3]
    r_w = quat_to_rotvec(quat_quat_mult(xyz_to_quat(x_c[3:]), q0))
    return ca.vertcat(x_w, r_w)

# Transform a full trajectory from compliance frame to world
def comp_traj_to_world(init_pose, traj):
    traj_world = np.zeros((6,traj.shape[1]))
    for i in range(traj.shape[1]):
        traj_world[:,i] = np.squeeze(comp_to_world(init_pose, traj[:,i]))
    return traj_world

def force_comp_to_world(rotvec, force_comp):
    rot = rotvec_to_rotation(rotvec)
    return rot.T @ force_comp[0:3]

