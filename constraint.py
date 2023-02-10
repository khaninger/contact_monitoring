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
    
    def fit(self, data):
        #self.max_dist(data)
        # in: data is the trajectory that we measure from the demonstration
        loss = 0
        for data_pt in data:
            loss += self.violation(data_pt)
        # get dec vars; x is symbolic decision vector of all params, lbx/ubx lower/upper bounds
        x, lbx, ubx = self.params.get_dec_vectors()
        x0 = self.params.get_x0()
        args = dict(x0=x0, lbx=lbx, ubx=ubx, p=None)
        prob = dict(f=loss, x=x)
        solver = ca.nlpsol('solver', 'ipopt', prob)
        sol = solver(x0 = x0, lbx = lbx, ubx = ubx)
        self.params.set_results(sol['x'])
        print(self.params)

        #self.build_jac()
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


    def violation(self, x):
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

class PointConstraint_v2(Constraint):
    # a point on the rigidly held object is fixed in world coordinates
    def __init__(self):
        params_init = {'radius_1': np.zeros(1),
                       'radius_2': np.zeros(1),
                       'radius_3': np.zeros(1),
                        'rest_pt': np.zeros(3)}

        Constraint.__init__(self, params_init)

    def violation(self, x):
        print("DEBUG!!!")
        print(x.shape)
        loss = ca.fabs(self.params['radius_1'] - ca.norm_2(x[0, :3] - self.params['rest_pt']))
        loss += ca.fabs(self.params['radius_2'] - ca.norm_2(x[1, :3] - self.params['rest_pt']))
        loss += ca.fabs(self.params['radius_3'] - ca.norm_2(x[2, :3] - self.params['rest_pt']))
        loss += 0.1 * (self.params['radius_1']+self.params['radius_2']+self.params['radius_3'])
        return loss


class PointConstraint_v3(Constraint):
    # a point on the rigidly held object is fixed in world coordinates
    def __init__(self):
        #params_init = {'radius_1': np.array([0.28]),
         #               'rest_pt': np.array([-0.31187662, -0.36479221, -0.03707742])}

        params_init = {'radius_1': np.array([0]),
                       'rest_pt': np.array([0, 0, 0])}

        Constraint.__init__(self, params_init)

    def violation(self, x):
        print("DEBUG!!!")
        print(x.shape)
        loss = ca.fabs(self.params['radius_1'] - ca.exp(1+1000*ca.norm_2(x - self.params['rest_pt'])))
        loss += 0.1 * (self.params['radius_1'])
        return loss

class CableConstraint(Constraint):
    # a point is constrained to be a certain distance from a point in world coordinates
    def __init__(self, rest_pt = np.zeros(3)):
        params_init = {'rest_pt': rest_pt,
                       'radius': np.zeros(1)} # center of the sphere of motion
        Constraint.__init__(self, params_init)
        
    def violation(self, x):
        # in: x is a position w/ 3 elements
        if x.shape == (4,4): #if x is trans mat: extract point
            x = x[:3, 3]
        dist = ca.norm_2(x-self.params['rest_pt'])
        return ca.norm_2(dist-self.params['radius']) +  0.1 * self.params['radius']

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
        

class Line(Constraint):

    def __init__(self):
        params_init = {'pt_0': np.zeros(3), 'pt_1': np.zeros(3),  # contact point in the object frame, which changes wrt 'x'
                       'rest_plane': np.zeros((4,4))}  # resting pose of contact plane in world coordinates (z_axis is normal of plane pointing in direction of free motion)
        Constraint.__init__(self, params_init)

    def violation(self, x):
        x_pt_0 = x @ ca.vertcat(self.params['pt_0'], ca.SX(1))  # Transform 'pt' into world coordinates
        x_pt_1 = x @ ca.vertcat(self.params['pt_1'], ca.SX(1))  # Transform 'pt' into world coordinates
        # minimize distance to plane in z direction
        #   ???
        # -> punish close and large distances of pt_0 to pt_1
        # -> restrict rest_plane[:3,:3] to be a rotation matrix
        #   -> ca.norm_2(self.params['rest_pt'][:3,i])==1 for i = 0;1;2  with slack!?
        #   -> ca.mtimes(self.params['rest_pt'][:3,i],self.params['rest_pt'][:3,j])==0 for i,j = 0,1; 0,2; 1,2 with slack?
        # -> restrict rest_plane[3,:2] to be 0?
        
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
    from dataload_helper import point_c_data, plug
    from visualize import plot_3d_points, plot_3d_points_segments
    #pts = point_c_data(n_points=3, noise=0.01, rot_0=0.5, rot_1=1).points()
    pts = plug(index=3, segment=True).load()
    #print("pts.shape")
    #print(pts.shape)
    print("start fit")
    #constraint = PointConstraint_v2()
    constraint = PointConstraint_v3()
    params = constraint.fit(pts[1])
    plot_3d_points_segments(L=pts, rest_pt=params['rest_pt'])
    #plot_3d_points_segments(L=pts, rest_pt=np.array([-0.31187662, -0.36479221, -0.03707742]))
