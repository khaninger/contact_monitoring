import numpy as np
import casadi as ca
import matplotlib as plt
from rotation_helpers import rotvec_to_rotation

# cable constraint: a point is constrained to be a certain distance from a point in world coordinates

def h_cable(x):
    # IN: x is object pose
    rest_pt = np.array([0,0,0])  # rest position
    radius = 15  # [cm]
    if x.shape == (4,4):
        x = x[:3,3]
    dist = ca.norm_2(x-rest_pt)
    violation = ca.norm_2(dist-radius +0.1*radius)
    return violation

# Line on surface constraint: a line on the object is flush on a surface, but is free to rotate about that surface

def line_on_surface(x):
    # IN: x is a pose for the object --> (4,4) transformation matrix
    surf_normal = np.array([0,0,1])
    surf_height = 2
    line_pt_on_obj = np.array([0,0,0])
    line_orient = np.array([1,0,0])
    line_in_space = x @ ca.vertcat(line_orient, ca.SX(1))
    misalign = line_in_space[:3].T@surf_normal  # these should be orthogonal, so dot product is zero
    line_pt_world = x @ ca.vertcat(line_pt_on_obj, ca.SX(1))
    displace = ca.abs(surf_normal.T@line_pt_world - surf_height)
    displace /= ca.norm_2(line_pt_on_obj)
    return displace + misalign

def double_point_constraint(x):
    pt_0 = np.array([0,0,1])
    pt_1 = np.array([1,0,0])
    plane_normal = np.array([1,0,0])
    plane_contact = 0.5
    ca.fabs(ca.norm_2(pt_0 - pt_1) - .1)  # Distance of both contact points is 10cm
    ca.norm2(plane_normal - 1)  # length of norm is one
    x_pt_0 = x @ ca.vertcat(pt_0, ca.SX(1))[:3]
    x_pt_1 = x @ ca.vertcat(pt_1, ca.SX(1))[:3]
    loss = ca.vertcat(
        ca.cross(x_pt_0 - plane_contact*plane_normal, plane_normal),
        ca.cross(x_pt_1 - plane_contact * plane_normal, plane_normal)
    )
    return loss

def get_jacobian(constraint_type):
    x_sym = ca.SX.sym('x_sym', 6)
    R_sym = rotvec_to_rotation(x_sym[3:])
    rot = ca.vertcat(R_sym, ca.SX(1,3))
    pos = ca.vertcat(x_sym[:3], ca.SX(1))
    T_sym = ca.horzcat(rot,pos)
    h = constraint_type(T_sym)
    jac = ca.jacobian(h,x_sym)
    jac_fn = ca.Function('jac_fn', [x_sym], [jac])
    jac_pinv = ca.pinv(jac)
    jac_pinv_fn = ca.Function('jac_pinv_fn', [x_sym], [jac_pinv])

    return jac_fn, jac_pinv_fn








