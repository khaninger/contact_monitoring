from collections import deque
import casadi as ca
import numpy as np
import pickle

from .decision_vars import DecisionVar, DecisionVarSet
from .rotation_helpers import *


class Constraint():
    def __init__(self, params_init):
        # IN: params_init is a dictionary which sets the parameters to be optimized, their dimensions, and their initial values
        # IN: skip_opt loads the initial params as the final params, skipping the optimization
        self.params = DecisionVarSet(x0 = params_init) # builds a dec var set with init value x0, optional params xlb xub
        self.linear = False  # indicates that the constraint only depends on linear translation, not orientation
        self.T_final = None   # final pose in the training data for this constraint

    def set_params(self, params_init):
        # IN: params_init is a dictionary of the form of self.params
        print(f"Skipping fit on a {str(type(self))} \n-> just setting params")
        print(params_init)
        self.params = params_init
        self.get_jac()

    def fit(self, data, h_inf = True):
        # IN: data is the trajectory that we measure from the demonstration
        # IN: h_inf activates the hinf penalty and inequality constraints in the optimization problem
        print(f"Fitting {str(type(self))} \n-> with following params:")
        print(self.params)

        loss = 0
        ineq_constraints = []

        for data_pt in data:
            loss += ca.norm_2(self.violation(data_pt))
        loss += data.shape[0]*self.regularization()

        if h_inf:  # add a slack variable which will bound the violation, and be minimized
            self.params['slack'] = DecisionVar(x0 = [0.01], lb = [0.0], ub = [0.1])
            loss += data.shape[0]*self.params['slack']
            ineq_constraints = [ca.fabs(self.violation(d))-self.params['slack'] for d in data]

        # get dec vars; x is symbolic decision vector of all params, lbx/ubx lower/upper bounds
        x, lbx, ubx = self.params.get_dec_vectors()
        x0 = self.params.get_x0()
        args = dict(x0=x0, lbx=lbx, ubx=ubx, p=None, lbg=-np.inf, ubg=np.zeros(len(ineq_constraints)))

        prob = dict(f=loss, x=x, g=ca.vertcat(*ineq_constraints))
        solver = ca.nlpsol('solver', 'ipopt', prob, {'ipopt.print_level':5})

        # solve, print and return
        sol = solver(x0 = x0, lbx = lbx, ubx = ubx)
        self.params.set_results(sol['x'])
        self.get_jac()

        self.params['T_traj'] = data        # save the full trajecory
        self.params['T_final'] = data[-1]   # save the final point in the dataset
        print(f"Optimized params: \n {self.params}")

        return self.params

    def violation(self, T): # constraint violation for a transformation matrix T
        raise NotImplementedError

    def get_jac(self): # construct the jacobian and pinv
        x_tcp_sym = ca.SX.sym("x_tcp_sym",3+3*(not self.linear))
        T_tcp_sym = self.pose_to_tmat(x_tcp_sym)
        T_obj_sym = T_tcp_sym
        #T_obj_sym = (Transformation tcp to object)@T_tcp_sym
        h = self.violation(T_obj_sym)
        self.jac = ca.jacobian(h, x_tcp_sym)
        self.jac_fn = ca.Function('jac_fn', [x_tcp_sym], [self.jac])
        self.jac_pinv = ca.pinv(self.jac)
        self.jac_pinv_fn = ca.Function('jac_pinv_fn', [x_tcp_sym], [self.jac_pinv])

    def calc_constraint_wrench(self, T, magnitude):
        # IN: T is the pose of the object in transformation matrix
        # IN: magniutude is the amount of force in that direction
        contact_jac = self.jac_fn(self.tmat_to_pose(T))
        wrench = ca.SX.zeros(6)
        for row in range(contact_jac.shape[0]):
            jac_element = contact_jac[row,:3]
            wrench[:3] += magnitude*jac_element.T/ca.norm_2(jac_element)
        
        return wrench

    def get_similarity(self, T, f):
        # IN: T is a transformation matrix for pose
        # IN: f is the numerical value for measured force
        x = self.tmat_to_pose(T)
        #print(f'in get_sim linear val: {self.linear} with type {type(self)}')
        if self.linear:
            f = f[:3]
        #print(f'jacobian: {self.jac_fn(x)}')
        #print(f'f: {ca.SX(f).shape}') 6,1
        #print(f'jac: {self.jac_fn(x).shape}') 1,6
        #print(f'pinv: {self.jac_pinv_fn(x).shape}') 6,1
        #return ca.norm_2(ca.SX(f)-self.jac_fn(x).T@(ca.SX(f).T@self.jac_pinv_fn(x)))
        return ca.norm_2(ca.SX(f)-self.jac_pinv_fn(x)@(self.jac_fn(x))@ca.SX(f))

    def pose_to_tmat(self, x): # x is the pose representation
        if self.linear:
            x_aug = ca.vertcat(x, ca.DM.zeros(3))
            T = xyz_pose_to_tmat(x_aug)
        else:
            T = xyz_pose_to_tmat(x)
        return T

    def tmat_to_pose(self, T): # T is the transformation matrix
        if self.linear:
            return T[:3,-1]
        else:
            return tmat_to_xyz_pose(T)

    def save(self):
        return (type(self), self.params)

class FreeSpace(Constraint):
    def __init__(self):
        Constraint.__init__(self, {})

    def fit(self, data):
        self.params['T_traj'] = [data[-1]]
        self.params['T_final'] = data[-1]
        print(f"Optimized params: \n {self.params}")
        return self.params

    def violation(self, T):
        return 0.0

class PointConstraint(Constraint):
    # a point on the rigidly held object is fixed in world coordinates
    def __init__(self):
        params_init = {'pt': np.zeros(3),       # contact point in the object frame, which changes wrt 'x'
                       'rest_pt': np.zeros(3)} # resting position of the point contact in world coordinates
        Constraint.__init__(self, params_init)

    def violation(self, T):
        # in: T is the object pose in a 4x4 transformation matrix
        x_pt = transform_pt(T, self.params['pt'])  # Transform 'pt' into world coordinates
        return ca.norm_2(x_pt - self.params['rest_pt'])

class CableConstraint(Constraint):
    # a point on the rigidly held object is fixed in world coordinates
    def __init__(self):
        params_init = {'radius_1': np.array([0.5]),
                       'rest_pt':  np.array([0, 0, 0])}
        Constraint.__init__(self, params_init)
        self.linear = True  # flag variable to switch between full jacobian and linear one

    def regularization(self):
        return 0.001 * self.params['radius_1']

    def violation(self, T):
        x = self.tmat_to_pose(T)
        return self.params['radius_1'] - ca.norm_2(x - self.params['rest_pt'])


class LineOnSurfaceConstraint(Constraint):
    # A line on the object is flush on a surface, but is free to rotate about that surface
    def __init__(self):
        params_init = {'surf_normal': np.zeros(3), # Normal vector of surface
                       'surf_height': np.zeros(1), # Height of surface in global coord
                       'line_pt_on_obj': np.zeros(3), # A point on the line
                       'line_orient': np.zeros(3), # Direction of line in object coord
                       }
        Constraint.__init__(self, params_init)

    def violation(self, T):
        # in: T is a tmat for the object
        # alignment error
        line_in_space = transform_pt(T,self.params['line_orient'])
        misalign = line_in_space.T@self.params['surf_normal'] #these should be orthogonal, so dot product zero

        # displacement error between point on line and plane
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_plane 'hyperplane and arbitary point'
        line_pt_in_world = transform_pt(T, self.params['line_pt_on_obj'])
        displace = ca.abs(self.params['surf_normal'].T@line_pt_in_world - self.params['surf_height'])
        displace /= ca.norm_2(self.params['line_pt_on_obj'])

        return ca.vertcat(misalign,displace)

class RakeConstraint_1pt(Constraint):
    def __init__(self):
        params_init = {'pt': np.array([0.1, 0, 0]),  # first contact point in the object frame, which changes wrt 'x'
                       # second contact point in the object frame, which changes wrt 'x'
                       'plane_normal': np.array([0, 1, 0]),
                       'd': np.array([1])}  # resting position of the point contact in world coordinates

        Constraint.__init__(self, params_init)

    def regularization(self):
        return 0.001*ca.fabs(ca.norm_2(self.params['plane_normal']) - 1) + \
                0.00001 * ca.norm_2(self.params['pt'])
        # length of norm is one

    def violation(self, T):
        x_pt = transform_pt(T, self.params['pt'])
        plane_normal = self.params['plane_normal'] / ca.norm_2(self.params['plane_normal'])
        delta_pt = ca.fabs(ca.dot(x_pt, plane_normal) - self.params['d'])
        return ca.vertcat(delta_pt, delta_pt)




class RakeConstraint_2pt(Constraint):
    def __init__(self):
        params_init = {'pt_0': np.array([0.01, 0, 0]),  # first contact point in the object frame, which changes wrt 'x'
                       'pt_1': np.array([-0.01, 0, 0]),# second contact point in the object frame, which changes wrt 'x'
                       'plane_normal': np.array([0, 1, 0]),
                       'd': np.array([1])}  # resting position of the point contact in world coordinates

        Constraint.__init__(self, params_init)

    def regularization(self):
        return 0.001*(10*ca.fabs(ca.norm_2(self.params['plane_normal']) - 1) +
                      ca.norm_2(self.params['pt_0']) +
                      ca.norm_2(self.params['pt_1']) +
                      ca.norm_2(self.params['pt_0']-self.params['pt_1'])-0.2)
        # length of norm is one

    def violation(self, T):
        x_pt0 = transform_pt(T, self.params['pt_0'])
        x_pt1 = transform_pt(T, self.params['pt_1'])
        plane_normal = self.params['plane_normal']
        delta_x = ca.fabs(ca.dot((x_pt0 - x_pt1), plane_normal))
        delta_0 = ca.dot(x_pt0, plane_normal) - self.params['d']
        delta_1 = ca.dot(x_pt1, plane_normal) - self.params['d']
        return ca.vertcat(delta_0 + delta_x, delta_1 + delta_x)

class RakeConstraint_Hinge(Constraint):
    def __init__(self):
        params_init = {'pt_0': np.array([0.01, 0, 0]),  # first contact point in the object frame, which changes wrt 'x'
                       'pt_1': np.array([-0.01, 0, 0]),# second contact point in the object frame, which changes wrt 'x'
                       'rest_pt_0': np.array([1, 0, 0]),
                       'rest_pt_1': np.array([-1, 0, 0]),
                       }  # resting position of the point contact in world coordinates

        Constraint.__init__(self, params_init)

    def regularization(self):
        return 0.01*(
                    ca.norm_2(self.params['pt_0']) +
                      ca.norm_2(self.params['pt_1']) +
                      ca.fabs(ca.norm_2(self.params['pt_0']-self.params['pt_1'])-0.2))
        # length of norm is one

    def violation(self, T):
        x_pt0 = transform_pt(T, self.params['pt_0'])
        x_pt1 = transform_pt(T, self.params['pt_1'])
        delta_0 = ca.norm_2(self.params['rest_pt_0'] - x_pt0)
        delta_1 = ca.norm_2(self.params['rest_pt_1'] - x_pt1)
        return ca.vertcat(delta_0, delta_1)

class ConstraintSet():
    def __init__(self, file_path = None):
        # IN: if file_path, will load constraint set from there
        self.constraints = {}
        self.sim_score = {}
        self.jac = {}
        self.vio = {}
        self.force_buffer = deque(maxlen=8)
        self.con_buffer = deque(maxlen=1)  # storing actual and previous constraint for smoothing

        if file_path:
            self.load(file_path)

    def fit(self, names, constraints, datasets):
        # in: datasets is a list of clustered data, in order.
        # in: constraints is a list of constraint objects, in order.
        for name, const_type, data in zip(names, constraints, datasets):
            const = const_type()
            const.fit(data)
            self.constraints[name] = const

    def load(self, file_path): # load fit params, and create the set of constraints
        save_dict = pickle.load(open(file_path, 'rb'))# load pickle file
        for name in save_dict.keys():
            print(f"loading constraint: {name}")
            # first element in tuple in save_dict[name] is the type, 2nd is the params
            # we are calling the type to construct the object, with argument of the params
            const = save_dict[name][0]()
            const.set_params(save_dict[name][1])
            self.constraints[name] = const

    def save(self, file_path):
        # generate a dict with the keys from contraint
        save_dict = {name: self.constraints[name].save() for name in self.constraints.keys()}
        pickle.dump(save_dict, open(file_path, 'wb'))

    def get_next(self, constraint):
        # IN: the constraint to get a successor of
        constraints_list = list(self.constraints.values())
        index = constraints_list.index(constraint)+1
        if index < len(constraints_list):
            return constraints_list[index]
        else:
            return None

    def id_constraint(self, x, f):
        # identify which constraint is most closely matching the current force
        threshold = 6   # 6 for cable, 2.5 for rake
        tol_violation = 0.5  # to be defined
        tol_sim = 2.5  # to be better defined
        self.force_buffer.append(np.linalg.norm(f))
        for name, constr in self.constraints.items():
            self.sim_score[name] = constr.get_similarity(x, f)
            #if name != 'free_space': print(f'{name}: {constr.jac_fn(constr.tmat_to_pose(x))}')
            self.vio[name] = constr.violation(x)


        #print(f"Sim score: {self.sim_score}")
        if (any(it<threshold for it in self.force_buffer)): # or (all(itr>tol_violation for itr in self.vio.values())):
            print('\rFree space               ', end="")
            active_con = 'free_space'
            #print(self.force_buffer)
        else:
            new_con = min(self.sim_score, key=lambda  y: self.sim_score[y])
            if len(self.con_buffer) is 0 or self.sim_score[new_con] < self.sim_score[self.con_buffer[0]] - tol_sim: # We accept the new constraint because it's better
                print(f'\nConstraint: {new_con}                  ')
                active_con = new_con
                self.con_buffer.append(new_con)  # Save it for future comparison
            else: # We reject the new constraint because its not better
                #print(f"Sim score: {self.sim_score}")
                active_con = self.con_buffer[0]
        self.con_buffer.append(active_con)
        is_final = True if (active_con == list(self.constraints)[-1]) else False
        return self.sim_score, self.constraints[active_con], active_con


if __name__ == "__main__":
    print("try test.py instead :)")
