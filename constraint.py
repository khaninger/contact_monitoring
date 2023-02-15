import casadi as ca
import numpy as np
from decision_vars import DecisionVar, DecisionVarSet

from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

class Constraint():
    def __init__(self, params_init):
        # in: params_init is a dictionary which sets the parameters to be optimized, their dimensions, and their initial values 
        self.params = DecisionVarSet(x0 = params_init) # builds a dec var set with init value x0, optional params xlb xub
        print("Initializing a Constraint with following params:")
        print(self.params)
    
    def fit_h2(self, data):
        #self.max_dist(data)
        # in: data is the trajectory that we measure from the demonstration
        loss = 0
        for data_pt in data:
            loss += ca.norm_2(self.violation(data_pt))

        loss += data.shape[0]*self.regularization()
        # get dec vars; x is symbolic decision vector of all params, lbx/ubx lower/upper bounds
        x, lbx, ubx = self.params.get_dec_vectors()
        x0 = self.params.get_x0()
        args = dict(x0=x0, lbx=lbx, ubx=ubx, p=None)
        prob = dict(f=loss, x=x)
        solver = ca.nlpsol('solver', 'ipopt', prob)
        sol = solver(x0 = x0, lbx = lbx, ubx = ubx)
        self.params.set_results(sol['x'])
        print(self.params)
        return self.params
 
    def fit_hinf(self, data):
        # in: data is the trajectory that we measure from the demonstration

        # add a slack variable which will bound the violation, and be minimized
        self.params['slack'] = DecisionVar(x0 = [0.5], lb = [0.0], ub = [0.01])
        loss = 0
        for d in data:
            loss += ca.norm_2(self.violation(d))# - 10*self.params['slack']
        loss += data.shape[0]*(self.regularization() + self.params['slack'])

        # make the inequality constraints, which should always be <0, i.e. |violation|<slack
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
        return self.params

    def max_dist(self, data):
        #https://stackoverflow.com/questions/31667070/max-distance-between-2-points-in-a-data-set-and-identifying-the-points
        if data[0].shape == (4,4):
            datapoints = data[:, :3, 3]
        else:
            datapoints = data
        # Returned 420 points in testing
        hull = ConvexHull(datapoints)

        # Extract the points forming the hull
        hullpoints = datapoints[hull.vertices, :]
        # Naive way of finding the best pair in O(H^2) time if H is number of points on
        # hull
        hdist = cdist(hullpoints, hullpoints, metric='euclidean')
        # Get the farthest apart points
        bestpair = np.unravel_index(hdist.argmax(), hdist.shape)

        self.data_maxd = np.linalg.norm(hullpoints[bestpair[0]] - hullpoints[bestpair[1]])


    def violation(self, x: object) -> object:
        # constraint violation for a single pose x
        raise NotImplementedError

    def build_jac(self, x):
        # jacobian evaluated at point x
        #x_sym for pose
        #h = violation(x_sym)
        #self.jac_fn = ca.jacobian(h, x_sym)
        #self.jac_pinv_fn = ca.pinv(...)
        raise NotImplementedError

    def get_similarity(self, x, f):
        # IN: x is a numerical value for a pose
        # IN: f is the numerical value for measured force
        # TODO add the least squares residual calculation
        # TODO check x and f are same dim
        assert (x.shape == f.shape)
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
                       'd': np.array([.1])}  # resting position of the point contact in world coordinates

        Constraint.__init__(self, params_init)

    def regularization(self):
        return (ca.fabs(
            ca.norm_2(self.params['pt_0'] - self.params['pt_1']) - .1)  # Distance of both contact points is 10cm
                + ca.fabs(ca.norm_2(self.params['plane_normal']) - 1))  # length of norm is one

    def violation(self, x):
        x_pt_0 = (x @ ca.vertcat(self.params['pt_0'], ca.SX(1)))[:3]
        x_pt_1 = (x @ ca.vertcat(self.params['pt_1'], ca.SX(1)))[:3]
        # dot product of plane normal and pt_i - plane_contact = 0
        loss = ca.vertcat(
            x_pt_0[0] * self.params['plane_normal'][0] +
            x_pt_0[1] * self.params['plane_normal'][1] +
            x_pt_0[2] * self.params['plane_normal'][2] - self.params['d'],
            x_pt_1[0] * self.params['plane_normal'][0] +
            x_pt_1[1] * self.params['plane_normal'][1] +
            x_pt_1[2] * self.params['plane_normal'][2] - self.params['d'],
        )
        return loss


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

    cable = False

    if cable:
        pts_1 = plug(index=1, segment=True).load()[1]
        pts_2 = plug(index=2, segment=True).load()[1]
        pts_3 = plug(index=3, segment=True).load()[1]
        pts_L = [pts_1, pts_2, pts_3]
        pts= np.vstack((pts_1, pts_2, pts_3))
        constraint = CableConstraint()
        #params = constraint.fit_h2(pts)
        #params = constraint.fit_hinf(pts_3)
        #plot_3d_points_segments(L=[pts_3], radius=params['radius_1'], rest_pt=params['rest_pt'])

        for i in range(3):
            constraint = CableConstraint()
            params = constraint.fit_hinf(pts_L[i])
            print(f"** Ground truth **\nradius_1:\n: {0.28}\nrest_pt:\n: {np.array([-0.31187662, -0.36479221, -0.03707742])}")
            plot_3d_points_segments(L=[pts_L[i]], radius=params['radius_1'] , rest_pt=params['rest_pt'], exp_n=i+1)

        #plot_3d_points_segments(L=pts, rest_pt=params['rest_pt'])
    else:
        pts_1, segments, time = rake(index=5, segment=True).load(pose=True)
        #pts_2 = rake(index=2, segment=True).load(pose=True)[1]
        #pts_3 = rake(index=3, segment=True).load(pose=True)[1]
        #pts_4 = rake(index=4, segment=True).load(pose=True)[1]
        #pts_5 = rake(index=5, segment=True).load(pose=True)[1]

        pts = pts_1[1]
        print(pts.shape)

        constraint = DoublePointConstraint()
        #constraint = DoublePointConstraint2()
        params = constraint.fit_hinf(pts)
        #params = constraint.fit_h2(pts)
        L_pt = [params['pt_0'], params['pt_1']]
        plane = [params['pt_0'], params['d']]
        plot_x_pt_inX(L_pt=L_pt, X=pts)


