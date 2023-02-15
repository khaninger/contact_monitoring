import casadi as ca
import numpy as np
from decision_vars import DecisionVar, DecisionVarSet

from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from rotation_helpers import rotvec_to_rotation

class Constraint():
    def __init__(self, params_init):
        # in: params_init is a dictionary which sets the parameters to be optimized, their dimensions, and their initial values 
        self.params = DecisionVarSet(x0 = params_init) # builds a dec var set with init value x0, optional params xlb xub
        print("Initializing a Constraint with following params:")
        print(self.params)
    
    def fit(self, data, h_inf = True):
        # in: data is the trajectory that we measure from the demonstration
        # in: h_inf activates the hinf penalty and inequality constraints in the optimization problem
        loss = 0
        ineq_constraints = []

        for data_pt in data:
            loss += ca.norm_2(self.violation(data_pt))

        loss += data.shape[0]*self.regularization()

        if h_inf:
            # add a slack variable which will bound the violation, and be minimized
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
        #self.get_jac()
        return self.params
 
    def violation(self, x: object) -> object:
        # constraint violation for a single pose x
        raise NotImplementedError

    def get_jac(self):
        # jacobian constructed at x_sym
        if not hasattr(self,"jac_fn"):
            x_sym = ca.SX.sym("x_sym",6)
            R_sym = rotvec_to_rotation(x_sym[3:])
            rot = ca.vertcat(R_sym, ca.SX(1,3))
            pos = ca.vertcat(x_sym[:3], ca.SX(1))
            T_sym = ca.horzcat(rot,pos)  # simbolic transformation matrix
            h = self.violation(T_sym)
            self.jac = ca.jacobian(h, x_sym)
            self.jac_fn = ca.Function('self.jac_fn', [x_sym], [self.jac])
            self.jac_pinv = ca.pinv(self.jac)
            self.jac_pinv_fn = ca.Function('self.jac_pinv_fn', [x_sym], [self.jac_pinv])

        return self.jac_fn, self.jac_pinv_fn

    def get_similarity(self, x, f):
        # IN: x is a numerical value for a pose
        # IN: f is the numerical value for measured force
        # TODO add the least squares residual calculation
        # TODO check x and f are same dim

        assert (x.shape == f.shape)
        jac_fn = self.get_jac()
        self.jac_fn()
        return ca.norm_2(f-self.jac_fn(x).T@(f@self.jac_pinv_fn(x))).full()
        raise NotImplementedError

class PointConstraint(Constraint):
    # a point on the rigidly held object is fixed in world coordinates
    def __init__(self):
        params_init = {'pt': np.zeros(3),       # contact point in the object frame, which changes wrt 'x'
                       'rest_pt': np.zeros(3)} # resting position of the point contact in world coordinates
        Constraint.__init__(self, params_init)
        
    def violation(self, x):
        # in: x is the object pose in a 4x4 transformation matrix
        # if h(x)=0 represents the constraint, violation is returning  h(x)
        x_pt = x @ ca.vertcat(self.params['pt'], ca.SX(1))  # Transform 'pt' into world coordinates
        return ca.norm_2(x_pt[:3]-self.params['rest_pt'])

class CableConstraint(Constraint):
    # a point on the rigidly held object is fixed in world coordinates
    def __init__(self):

        params_init = {'radius_1': np.array([0.5]),
                       'rest_pt': np.array([0, 0, 0])}

        #params_init = {'rest_pt': np.array([0, 0, 0])}
        Constraint.__init__(self, params_init)
        #self.params['radius_1'] = DecisionVar(x0=[0.5], lb=[0.0], ub=[10])
    def regularization(self):
        return 0.001 * self.params['radius_1']

    def violation(self, x):
        #return self.params['radius_1'] - ca.exp(1+ca.norm_2(x - self.params['rest_pt']))
        return self.params['radius_1'] - ca.norm_2(x - self.params['rest_pt'])

    def violation2(self, x):
        return self.params['radius_1'] - ca.exp(1+10*ca.norm_2(x - self.params['rest_pt']))

class LineOnSurfaceConstraint(Constraint):
    # A line on the object is flush on a surface, but is free to rotate about that surface
    # 
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
        return ca.fabs(ca.norm_2(self.params['pt_0'] - self.params['pt_1']) - .1)\
            +  ca.fabs(ca.norm_2(self.params['plane_normal']) - 1)
        # length of norm is one

    def violation(self, x):
        x_pt_0 = (x @ ca.vertcat(self.params['pt_0'], ca.SX(1)))[:3]
        x_pt_1 = (x @ ca.vertcat(self.params['pt_1'], ca.SX(1)))[:3]
        plane_normal = self.params['plane_normal'] / ca.norm_2(self.params['plane_normal'])
        delta_pt0 = ca.fabs(ca.dot(x_pt_0, plane_normal) - self.params['d'])
        delta_pt1 = ca.fabs(ca.dot(x_pt_1, plane_normal) - self.params['d'])
        return ca.vertcat(delta_pt0, delta_pt1)



class ConstraintSet():
    def __init__(self, dataset):
        clusters = self.cluster(dataset)
        self.constraints = [] # list of constraints
        for cluster in clusters:
            c_type = self.id_constraint_pos(cluster)
            c_fit = self.fit_constraint(cluster, c_type)
            self.constraints.add(c_fit)
        
    def cluster(self, dataset):
        clusters = [] # list of partitioned data
        return clusters

    def fit_constraint(self, data, c_type):
        pass
    
    def id_constraint_pos(self, x):
        # identify which constraint type is most likely given the pose
        pass
    
    def id_constraint_force(self, x, f):
        # identify which constraint is most closely matching the current force
        for constr in self.constraints:
            sim_score = constr.get_similarity(x, F)
        pass

if __name__ == "__main__":
    from dataload_helper import point_c_data, plug, rake
    from visualize import plot_x_pt_inX, plot_3d_points_segments

    cable = True

    if cable:
        pts_1 = plug(index=1, segment=True).load()[1]
        pts_2 = plug(index=2, segment=True).load()[1]
        pts_3 = plug(index=3, segment=True).load()[1]
        pts_L = [pts_1, pts_2, pts_3]

        for i in range(3):
            ind = i+1
            pts = plug(index=ind, segment=True).load()[1]
            constraint = CableConstraint()
            params = constraint.fit(pts_3, h_inf=True)
            plug().save(rest_pt=params['rest_pt'], radius=params['radius_1'], specifier='0', index=ind)

        """
        for i in range(3):
            constraint = CableConstraint()
            params = constraint.fit(pts_L[i], h_inf=True)
            print(f"** Ground truth **\nradius_1:\n: {0.28}\nrest_pt:\n: {np.array([-0.31187662, -0.36479221, -0.03707742])}")
            plot_3d_points_segments(L=[pts_L[i]], radius=params['radius_1'] , rest_pt=params['rest_pt'], exp_n=i+1)
        """

        #plot_3d_points_segments(L=pts, rest_pt=params['rest_pt'])
    else:
        pose_data, segments, time = rake(index=3, segment=True).load(pose=True, kp_delta_th=0.005)
        kp_data, segments, time = rake(index=3, segment=True).load(pose=False, kp_delta_th=0.005)

        plot_x_pt_inX(L_pt=kp_data[1])

        #constraint = DoublePointConstraint()
        #params = constraint.fit(pose_data[1], h_inf=True)
        #L_pt = [params['pt_0'], params['pt_1']]
        #plane = [params['plane_normal'], params['d']]
        #plot_x_pt_inX(L_pt=L_pt, X=pose_data[1], plane=plane)


