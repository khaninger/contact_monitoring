import casadi as ca
import numpy as np
import pickle

from .decision_vars import DecisionVar, DecisionVarSet

from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from .rotation_helpers import *

class Constraint():
    def __init__(self, params_init):
        # IN: params_init is a dictionary which sets the parameters to be optimized, their dimensions, and their initial values
        # IN: skip_opt loads the initial params as the final params, skipping the optimization
        self.params = DecisionVarSet(x0 = params_init) # builds a dec var set with init value x0, optional params xlb xub
        print("Initializing a Constraint with following params:")
        print(self.params)
        self.linear = False  # indicates that the constraint only depends on linear translation, not orientation

    def set_params(self, params_init):
        # IN: params_init is a dictionary of the form of self.params
        print("Skipping Optimization and just init params")
        print(f"**Params**\n{params_init}")
        self.params = params_init
        self.get_jac()

    def fit(self, data, h_inf = True):
        # IN: data is the trajectory that we measure from the demonstration
        # IN: h_inf activates the hinf penalty and inequality constraints in the optimization problem
        loss = 0
        ineq_constraints = []

        for data_pt in data:
            loss += ca.norm_2(self.violation(data_pt))
        loss += data.shape[0]*self.regularization()

        if h_inf:  # add a slack variable which will bound the violation, and be minimized
            self.params['slack'] = DecisionVar(x0 = [0.5], lb = [0.0], ub = [0.1])
            loss += data.shape[0]*self.params['slack']
            ineq_constraints = [ca.fabs(self.violation(d))-self.params['slack'] for d in data]

        # get dec vars; x is symbolic decision vector of all params, lbx/ubx lower/upper bounds
        x, lbx, ubx = self.params.get_dec_vectors()
        x0 = self.params.get_x0()
        args = dict(x0=x0, lbx=lbx, ubx=ubx, p=None, lbg=-np.inf, ubg=np.zeros(len(ineq_constraints)))
        prob = dict(f=loss, x=x, g=ca.vertcat(*ineq_constraints))
        solver = ca.nlpsol('solver', 'ipopt', prob)

        # solve, print and return
        sol = solver(x0 = x0, lbx = lbx, ubx = ubx)
        self.params.set_results(sol['x'])
        print(self.params)
        self.get_jac()
        return self.params

    def violation(self, x): # constraint violation for a transformation matrix x
        raise NotImplementedError

    def get_jac(self): # construct the jacobian and pinv
        if self.linear:
            x_sym = ca.SX.sym("x_sym",3)
            x_sym_aug = ca.vertcat(x_sym, ca.DM.zeros(3))
            T_sym = rotvec_pose_to_tmat(x_sym_aug)
        else:
            x_sym = ca.SX.sym("x_sym",6)
            T_sym = rotvec_pose_to_tmat(x_sym)
        h = self.violation(T_sym)
        self.jac = ca.jacobian(h, x_sym)
        self.jac_fn = ca.Function('jac_fn', [x_sym], [self.jac])
        self.jac_pinv = ca.pinv(self.jac)
        self.jac_pinv_fn = ca.Function('jac_pinv_fn', [x_sym], [self.jac_pinv])

    def get_similarity(self, x, f):
        # IN: x is a numerical value for a pose
        # IN: f is the numerical value for measured force
        if self.linear == True:
            f = f[:3]
            x = x[:3,-1]
        else:
            x = tmat_to_rotvec_pose(x)
        return ca.norm_2(ca.SX(f)-self.jac_fn(x).T@(ca.SX(f).T@self.jac_pinv_fn(x)))

    def save(self):
        return (type(self), self.params)


class PointConstraint(Constraint):
    # a point on the rigidly held object is fixed in world coordinates
    def __init__(self):
        params_init = {'pt': np.zeros(3),       # contact point in the object frame, which changes wrt 'x'
                       'rest_pt': np.zeros(3)} # resting position of the point contact in world coordinates
        Constraint.__init__(self, params_init)

    def violation(self, x):
        # in: x is the object pose in a 4x4 transformation matrix
        print(x)
        x_pt = x @ ca.vertcat(self.params['pt'], ca.SX(1))  # Transform 'pt' into world coordinates
        return ca.norm_2(x_pt[:3]-self.params['rest_pt'])

class CableConstraint(Constraint):
    # a point on the rigidly held object is fixed in world coordinates
    def __init__(self):
        params_init = {'radius_1': np.array([0.5]),
                       'rest_pt': np.array([0, 0, 0])}
        Constraint.__init__(self, params_init)
        self.linear = True  # flag variable to switch between full jacobian and linear one

    def regularization(self):
        return 0.001 * self.params['radius_1']

    def violation(self, x):
        print(x)
        return self.params['radius_1'] - ca.norm_2(x[:3,-1] - self.params['rest_pt'])

    def violation2(self, x):
        return self.params['radius_1'] - ca.exp(1+10*ca.norm_2(x[:3,-1] - self.params['rest_pt']))

class LineOnSurfaceConstraint(Constraint):
    # A line on the object is flush on a surface, but is free to rotate about that surface
    def __init__(self):
        params_init = {'surf_normal': np.zeros(3), # Normal vector of surface
                       'surf_height': np.zeros(1), # Height of surface in global coord
                       'line_pt_on_obj': np.zeros(3), # A point on the line
                       'line_orient': np.zeros(3), # Direction of line in object coord
                       }
        Constraint.__init__(self, params_init)

    def violation(self, x):
        # in: x is a pose for the object
        # alignment error
        line_in_space = x @ ca.vertcat(self.params['line_orient'], ca.SX(1))
        misalign = line_in_space[:3].T@self.params['surf_normal'] #these should be orthogonal, so dot product zero

        # displacement error between point on line and plane
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_plane 'hyperplane and arbitary point'
        line_pt_in_world = x @ ca.vertcat(self.params['line_pt_on_obj'], ca.SX(1))
        displace = ca.abs(self.params['surf_normal'].T@line_pt_in_world - self.params['surf_height'])
        displace /= ca.norm_2(self.params['line_pt_on_obj'])

        return misalign+displace

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

    def violation(self, x):
        x_pt_0 = (x @ ca.vertcat(self.params['pt_0'], ca.SX(1)))[:3]
        x_pt_1 = (x @ ca.vertcat(self.params['pt_1'], ca.SX(1)))[:3]
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

    def violation(self, x):
        x_pt_0 = (x @ ca.vertcat(self.params['pt_0'], ca.SX(1)))[:3]
        x_pt_1 = (x @ ca.vertcat(self.params['pt_1'], ca.SX(1)))[:3]
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

    def violation(self, x):
        loss =  ca.fabs(self.params['radius_0_0'] - ca.norm_2(x[0, :3] - self.params['pt_0']))
        loss += ca.fabs(self.params['radius_1_0'] - ca.norm_2(x[1, :3] - self.params['pt_0']))
        loss += ca.fabs(self.params['radius_2_0'] - ca.norm_2(x[2, :3] - self.params['pt_0']))
        loss += ca.fabs(self.params['radius_3_0'] - ca.norm_2(x[3, :3] - self.params['pt_0']))
        loss += ca.fabs(self.params['radius_0_1'] - ca.norm_2(x[0, :3] - self.params['pt_1']))
        loss += ca.fabs(self.params['radius_1_1'] - ca.norm_2(x[1, :3] - self.params['pt_1']))
        loss += ca.fabs(self.params['radius_2_1'] - ca.norm_2(x[2, :3] - self.params['pt_1']))
        loss += ca.fabs(self.params['radius_3_1'] - ca.norm_2(x[3, :3] - self.params['pt_1']))
        return ca.vertcat(loss, loss)

class ConstraintSet():
    def __init__(self):
        self.constraints = {}

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
            # first element in tuple in save_dict[name] is the type, 2nd is the params
            # we are calling the type to construct the object, with argument of the params
            const = save_dict[name][0]()
            const.set_params(save_dict[name][1])
            self.constraints[name] = const

    def save(self, file_path):
        # generate a dict with the keys from contraint
        save_dict = {name: self.constraints[name].save() for name in self.constraints.keys()}
        pickle.dump(save_dict, open(file_path, 'wb'))

    def id_constraint_force(self, x, f):
        # identify which constraint is most closely matching the current force
        for constr in self.constraints:
            sim_score = constr.get_similarity(x, f)
        pass

if __name__ == "__main__":
    import os
    from .dataload_helper import data
    from .visualize import plot_x_pt_inX, plot_3d_points_segments

    from .controller import Controller

    cable = True

    if cable:
        for i in range(3):
            ind = i+1
            pts, _, _ = data(index=ind, segment=True, data_name='plug').load(pose=True, kp_delta_th=0.005)
            print(pts[1].shape)
            constraint = CableConstraint()
            params = constraint.fit(pts[1], h_inf=True)
            print(
            f"** Ground truth **\nradius_1:\n: {0.28}\nrest_pt:\n: {np.array([-0.31187662, -0.36479221, -0.03707742])}")

    else:
        params = {'rest_pt': np.array([0,0,0]), 'pt': np.array([0.5,0.0,0.0])}
        const = PointConstraint()
        const.set_params(params)
        test = np.array([0.01,1.01,0.01,0.01,-0.01,0.02])
        print(const.jac_fn(test))
        """
        # Faire le testing du save
        constraint = CableConstraint()
        try:
            cont = Controller(constraint)
            cont.loop()
        finally:
            cont.stop()


        names = ['cable_fixture', 'front_pivot']

        constraints = [CableConstraint,
                       CableConstraint]  # list of constraints
        datasets = [plug(index=i+1, segment=True).load()[1] for i in range(2)]

        c_set = ConstraintSet()

        #c_set.fit(names=names, constraints=constraints, datasets=datasets)

        path = os.getcwd() + "/contact_monitoring/data/cable_constraint.pickle"
        #c_set.save(file_path=path)
        c_set.load(file_path=path)
        for i in range(2): print(c_set.constraints)
        """





