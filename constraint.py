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
        solver = ca.nlpsol('solver', 'ipopt', prob, {'ipopt.print_level':2})

        # solve, print and return
        sol = solver(x0 = x0, lbx = lbx, ubx = ubx)
        self.params.set_results(sol['x'])
        self.get_jac()

        self.params['T_final'] = data[-1]
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

    def get_similarity(self, T, f):
        # IN: T is a transformation matrix for pose
        # IN: f is the numerical value for measured force
        x = self.tmat_to_pose(T)
        if self.linear:
            f = f[:3]
        return ca.norm_2(ca.SX(f)-self.jac_fn(x).T@(ca.SX(f).T@self.jac_pinv_fn(x)))

    def pose_to_tmat(self, x): # x is the pose representation
        if self.linear:
            x_aug = ca.vertcat(x, ca.DM.zeros(3))
            T = rotvec_pose_to_tmat(x_aug)
        else:
            T = rotvec_pose_to_tmat(x)
        return T
        
    def tmat_to_pose(self, T): # T is the transformation matrix
        if self.linear:
            return T[:3,-1]
        else:
            return tmat_to_rotvec_pose(T)
    
    def save(self):
        return (type(self), self.params)

class FreeSpace(Constraint):
    def __init__(self):
        Constraint.__init__(self, {})

    def fit(self, data):
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
        return 0.0001 * self.params['radius_1']

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

class RakeConstraint(Constraint):
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



class RakeConstraint2(Constraint):
    def __init__(self):
        params_init = {'pt_0': np.array([0.01, 0, 0]),  # first contact point in the object frame, which changes wrt 'x'
                       'pt_1': np.array([-0.01, 0, 0]),# second contact point in the object frame, which changes wrt 'x'
                       'plane_normal': np.array([0, 1, 0]),
                       'd': np.array([1])}  # resting position of the point contact in world coordinates

        Constraint.__init__(self, params_init)

    def regularization(self):
        return 0.001*(ca.fabs(ca.norm_2(self.params['plane_normal']) - 1) +
                      ca.norm_2(self.params['pt_0']) +
                      ca.norm_2(self.params['pt_1']) +
                      ca.norm_2(self.params['pt_0']-self.params['pt_1'])-0.02)
        # length of norm is one

    def violation(self, T):
        x_pt0 = transform_pt(T, self.params['pt_0'])
        x_pt1 = transform_pt(T, self.params['pt_1'])
        plane_normal = self.params['plane_normal'] / ca.norm_2(self.params['plane_normal'])
        delta_pt0 = ca.fabs(ca.dot(x_pt0, plane_normal) - self.params['d'])
        delta_pt1 = ca.fabs(ca.dot(x_pt1, plane_normal) - self.params['d'])
        #delta_x =   ca.fabs(ca.dot((x_pt0 - x_pt1), plane_normal))
        #return ca.vertcat(delta_pt0+delta_x, delta_pt1+delta_x)
        return ca.vertcat(delta_pt0, delta_pt1)



class RakeConstraint3(Constraint):
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


class RakeConstraint_pt1(Constraint):
    # a point on the rigidly held object is fixed in world coordinates
    def __init__(self, params):
        pt_1_init = np.copy(params['pt'])
        pt_1_init[0] += 0.02

        params_init = {'pt_1': pt_1_init}  # resting position of the point contact in world coordinates
        self.pt_0 = params['pt']
        self.plane_normal = params['plane_normal']
        self.d = params['d']

        Constraint.__init__(self, params_init)

    def regularization(self):
        return ca.fabs(ca.norm_2(self.pt_0 - self.params['pt_1']) - .02)
        # length of norm is one

    def violation(self, T):
        x_pt_0 = transform_pt(T, self.pt_0)
        x_pt_1 = transform_pt(T, self.params['pt_1'])
        delta = ca.dot(self.plane_normal, (x_pt_0-x_pt_1))
        return ca.vertcat(delta, delta)
        #return ca.vertcat(delta_ptpt, delta_ptpt)

class ConstraintSet():
    def __init__(self, file_path = None):
        # IN: if file_path, will load constraint set from there
        self.constraints = {}
        self.sim_score = {}
        self.jac = {}
        self.vio = {}
        self.force_buffer = deque(maxlen=8)
        self.con_buffer = deque(maxlen=2)  # storing actual and previous constraint for smoothing


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

    def id_constraint(self, x, f):
        # identify which constraint is most closely matching the current force

        # identify which constraint is most closely matching the current force
        threshold = 6
        tol_violation = 0.5  # to be defined
        tol_sim = 0.4  # to be better defined
        self.force_buffer.append(np.linalg.norm(f))
        for name, constr in self.constraints.items():
            self.sim_score[name] = constr.get_similarity(x, f)
            #self.jac[name] = constr.jac_fn(x[:3,-1])
            self.vio[name] = constr.violation(x)
        active_con = min(self.sim_score, key=lambda  y: self.sim_score(y))
        self.con_buffer.append(active_con)



        if (any(it<threshold for it in self.force_buffer)) or (all(itr>tol_violation for itr in self.vio.values())):
            print("Free-space")
            return constraints['free_space']
        elif self.sim_score[self.con_buffer[1]] < self.sim_score[self.con_buffer[0]] -tol_violation:
            print(f"Sim score: {self.sim_score}")
            return self.sim_score, self.constraints[self.con_buffer[1]]
        else:
            print(f"Sim score: {self.sim_score}")
            return self.sim_score, self.constraints[self.con_buffer[0]]



=======
        if (any(it<threshold for it in self.force_buffer)) or (all(itr>tol for itr in self.vio.values())):
            #print("Free-space")
            return self.sim_score, self.constraints['free_space']
        else:
            #print(f"Sim score: {self.sim_score}")
            active_con = min(self.sim_score, key=lambda y: self.sim_score[y])
            return self.sim_score, self.constraints[active_con]
>>>>>>> 7b0f9d090ab2c1e33d04b922073eaa1c5c7091e6


if __name__ == "__main__":
    print("try test.py instead :)")
