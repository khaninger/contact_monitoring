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
            self.params['slack'] = DecisionVar(x0 = [0.01], lb = [0.0], ub = [0.01])
            loss += data.shape[0]*self.params['slack']
            ineq_constraints = [ca.fabs(self.violation(d))-self.params['slack'] for d in data]

        print("DEBUG LOSS")
        print(loss.shape)
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
        print(f"Optimized params: \n {self.params}")

        T_final = data[-1]
        print(f"final pose is {self.T_final}")

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

    def fit(self):
        pass

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

class DoublePointConstraint(Constraint):
    # a point on the rigidly held object is fixed in world coordinates
    def __init__(self):
        params_init = {'pt_0': np.array([0.05,0,0]),  # first contact point in the object frame, which changes wrt 'x'
                       'pt_1': np.array([-0.05,0,0]),  # second contact point in the object frame, which changes wrt 'x'
                       'plane_normal': np.array([1,0,0]),
                       'd': np.array([.1])}  # resting position of the point contact in world coordinates

        Constraint.__init__(self, params_init)

    def regularization(self):
        return (ca.fabs(ca.norm_2(self.params['pt_0']-self.params['pt_1']) -.1)   # Distance of both contact points is 10cm
        + ca.fabs(ca.norm_2(self.params['plane_normal']) - 1))              # length of norm is one

    def violation(self, T):
        x_pt_0 = transform_pt(T, self.params['pt_0'])
        x_pt_1 = transform_pt(T, self.params['pt_1'])
        # dot product of plane normal and pt_i - plane_contact = 0
        point_on_plane = self.params['d'] * self.params['plane_normal']
        loss = ca.vertcat(
            ca.dot((x_pt_0 - point_on_plane), self.params['plane_normal']),
            ca.dot((x_pt_1 - point_on_plane), self.params['plane_normal'])
            )
        return loss


class DoublePointConstraint2(Constraint):
    # a point on the rigidly held object is fixed in world coordinates
    def __init__(self):
        params_init = {'pt_0': np.array([0.05, 0, 0]),  # first contact point in the object frame, which changes wrt 'x'
                       'pt_1': np.array([-0.05, 0, 0]),
                       # second contact point in the object frame, which changes wrt 'x'
                       'plane_normal': np.array([1, 0, 0]),
                       'd': np.array([1])}  # resting position of the point contact in world coordinates

        Constraint.__init__(self, params_init)

    def regularization(self):
        return ca.fabs(ca.norm_2(self.params['pt_0'] - self.params['pt_1']) - .20)\
            +  ca.fabs(ca.norm_2(self.params['plane_normal']) - 1)
        # length of norm is one

    def violation(self, T):
        x_pt_0 = transform_pt(T, self.params['pt_0'])
        x_pt_1 = transform_pt(T, self.params['pt_1'])
        plane_normal = self.params['plane_normal'] / ca.norm_2(self.params['plane_normal'])
        delta_pt0 = ca.fabs(ca.dot(x_pt_0, plane_normal) - self.params['d'])
        delta_pt1 = ca.fabs(ca.dot(x_pt_1, plane_normal) - self.params['d'])
        return ca.vertcat(delta_pt0, delta_pt1)

class DoublePointConstraint3(Constraint):
    # a point on the rigidly held object is fixed in world coordinates
    def __init__(self):
        params_init = {'pt_0': np.array([0.05, 0, 0]),  # first contact point in the object frame, which changes wrt 'x'
                       'pt_1': np.array([-0.05, 0, 0]), # second contact point in the object frame, which changes wrt 'x'
                       'plane_normal': np.array([1, 0, 0]), # position of the plane contact in world coordinates ax + bx + cx = d
                       'd': np.array([1]), # position of the plane contact in world coordinates
                       'radius_0_0': np.array([1]),
                       'radius_0_1': np.array([1]),
                       'radius_1_0': np.array([1]),
                       'radius_1_1': np.array([1]),
                       'radius_2_0': np.array([1]),
                       'radius_2_1': np.array([1]),
                       'radius_3_0': np.array([1]),
                       'radius_3_1': np.array([1]),
                       }
        Constraint.__init__(self, params_init)

    def regularization(self):
        return ca.fabs(ca.norm_2(self.params['pt_0'] - self.params['pt_1']) - .20)\
            +  ca.fabs(ca.norm_2(self.params['plane_normal']) - 1)
        # length of norm is one

    def violation(self, T):
        # @Kevin 19.2: There's got to be a scalable way to do this... maybe a list for radius_01 or something? 
        loss =  ca.fabs(self.params['radius_0_0'] - ca.norm_2(T[0, :3] - self.params['pt_0']))
        loss += ca.fabs(self.params['radius_1_0'] - ca.norm_2(T[1, :3] - self.params['pt_0']))
        loss += ca.fabs(self.params['radius_2_0'] - ca.norm_2(T[2, :3] - self.params['pt_0']))
        loss += ca.fabs(self.params['radius_3_0'] - ca.norm_2(T[3, :3] - self.params['pt_0']))
        loss += ca.fabs(self.params['radius_0_1'] - ca.norm_2(T[0, :3] - self.params['pt_1']))
        loss += ca.fabs(self.params['radius_1_1'] - ca.norm_2(T[1, :3] - self.params['pt_1']))
        loss += ca.fabs(self.params['radius_2_1'] - ca.norm_2(T[2, :3] - self.params['pt_1']))
        loss += ca.fabs(self.params['radius_3_1'] - ca.norm_2(T[3, :3] - self.params['pt_1']))
        return ca.vertcat(loss, loss) # @Kevin 19.2: hmm? why stack same loss?

class ConstraintSet():
    def __init__(self, file_path = None):
        # IN: if file_path, will load constraint set from there
        self.constraints = {}
        self.sim_score = {}
        self.jac = {}
        self.force_buffer = deque(maxlen=8)


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
        threshold = 6
        self.force_buffer.append(np.linalg.norm(f))
        for name, constr in self.constraints.items():
            self.sim_score[name] = constr.get_similarity(x, f)
            self.jac[name] = constr.jac_fn(x[:3,-1])
        if any(it<threshold for it in self.force_buffer):
            #print("Free-space")
            pass
        else:
            #print(f"Sim score: {self.sim_score}")
            pass
        return self.sim_score
        #print(f"jac: {self.jac}")
        #print(f"jac: {constr.jac_fn(x[:3,-1])}")

if __name__ == "__main__":
    from .dataload_helper import *
    #plug_threading
    constraint = CableConstraint()

    #load threading data
    dataset, segments, time = data(index=1, segment=True, data_name="plug_threading").load(pose=True, kp_delta_th=0.005)
    cable_fixture = dataset[1]
    dataset, segments, time = data(index=1, segment=True, data_name="plug_threading").load(pose=True, kp_delta_th=0.005)
    front_pivot = dataset[2]


    names = ['cable_fixture', 'front_pivot']

    constraints = [CableConstraint,
                   CableConstraint]

    datasets = [cable_fixture, front_pivot]

    c_set = ConstraintSet()
    c_set.fit(names=names, constraints=constraints, datasets=datasets)

    path = os.getcwd() + "/contact_monitoring/data/cable_constraint.pickle"
    c_set.save(file_path=path)
    c_set.load(file_path=path)
    print(c_set.constraints)
