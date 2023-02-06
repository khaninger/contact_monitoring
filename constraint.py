import casadi as ca
import numpy as np
from decision_vars import DecisionVar, DecisionVarSet

class Constraint():
    def __init__(self):
        self.params = None
    
    def fit(self, data):
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

    def violation(self, x):
        # constraint violation for a single pose x
        raise NotImplementedError
    
    def get_jac(self, x):
        # jacobian evaluated at point x
        #x_sym for pose
        #h = violation(x_sym)
        #ca.jacobian(h, x_sym)
        print("TEST")
        raise NotImplementedError       
         


    def get_similarity(self, x, f):
        raise NotImplementedError

class PointConstraint(Constraint):
    # a point on the rigidly held object is fixed in world coordinates
    def __init__(self):
        Constraint.__init__(self)
        params_init = {'pt': np.zeros(3),       # contact point in the object frame, which changes wrt 'x'
                       'rest_pt': np.zeros(3),} # resting position of the point contact in world coordinates

        self.params = DecisionVarSet(x0 = params_init) # builds a dec var set with init value x0, optional params xlb xub
        print("Initializing a PointConstraint with following params:")
        print(self.params)

    def violation(self, x):
        # in: x is the object pose in a 4x4 transformation matrix
        # if h(x)=0 represents the constraint, violation is returning  h(x)
        x_pt = x @ ca.vertcat(self.params['pt'], ca.SX(1)) # Transform 'pt' into world coordinates
        return ca.norm_2(x_pt[:3]-self.params['rest_pt'])

class CableConstraint(Constraint):
    # a point is constrained to be a certain distance from a point in world coordinates
    def __init__(self):
        Constraint.__init__(self)
        params_init = {'rest_pt': np.zeros(3),
                       'radius': np.zeros(1)} # center of the sphere of motion
        self.params = DecisionVarSet(x0 = params_init)
        print("Initializing a CableConstraint with following params:")
        print(self.params)
        
    def violation(self, x):
        # in: x is a position w/ 3 elements
        dist = ca.norm_2(x-self.params['rest_pt'])
        return ca.norm_2(dist-self.params['radius'])

class ConstraintSet():
    def __init__(self, dataset):
        clusters = self.cluster(dataset)
        self.constraints = []
        for cluster in clusters:
            c_type = self.id_constraint_pos(cluster)
            c_fit = self.fit_constraint(cluster, c_type)
            self.constraints.add(c_fit)
        
    def cluster(self, dataset):
        clusters = [] # list of partitioned data
        return clusters

    def fit_constraint(self, data, c_type):
        pass
    
    def id_constraint_pos(self, data):
        pass
    
    def id_constraint_force(self, x, f):
        pass
