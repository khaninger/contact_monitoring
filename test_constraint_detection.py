import numpy as np
import casadi as ca
import matplotlib as plt

from rotation_helpers import *
# Test threading cable constraint

def cable_threading(x):
    rest_pt_1 = np.array([0,0,0])
    rest_pt_2 = np.array([10,0,0])
    radius = [5,5]
    violation = []
    rest = [rest_pt_1,rest_pt_2]
    for i in range(len(rest)):
        violation.append(radius[i]-ca.norm_2(x-rest[i]))

    return violation[0], violation[1]

def build_jac(str):
    x_sym = ca.SX.sym('x_sym',3)
    if str == '0':
        h = cable_threading(x_sym)[0]
    elif str == '1':
        h = cable_threading(x_sym)[1]
    else:
        print('No constraints in library')
    jac = ca.jacobian(h,x_sym)
    jac_fn = ca.Function('jac_fn', [x_sym], [jac])
    jac_pinv = ca.pinv(jac)
    jac_pinv_fn = ca.Function('jac_pinv_fn', [x_sym], [jac_pinv])
    return jac_fn, jac_pinv_fn

def similarity(x,f,flag):

    # IN: x is a numerical value for key-point coordinates
    # IN: f is a numerical value for force
    if flag == '0':
        jac_fn = build_jac('0')[0]
        jac_pinv_fn = build_jac('0')[1]
    elif flag == '1':
        jac_fn = build_jac('1')[0]
        jac_pinv_fn = build_jac('1')[1]

    return ca.norm_2(ca.SX(f)-jac_fn(x).T@(ca.SX(f).T @jac_pinv_fn(x)))

# Simulation

constraint_modes = ['0','1']
input_x = np.array([0,20,0])
input_force = np.array([0.2,-4.5,0])
jac = build_jac('0')[0]
jac1 = build_jac('1')[0]
print(jac(input_x),jac1(input_x))

metric = []
threshold = np.random.normal(0,0.4)
for i in constraint_modes:
    if np.linalg.norm(input_force)<threshold:
        print('Free-space')
    else:
        metric.append(similarity(input_x,input_force,i))
print(metric)
























