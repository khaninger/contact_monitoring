import os
from .dataload_helper import data
from .visualize import plot_x_pt_inX, plot_3d_points_segments
from .constraint import *
#from .controller import Controller

def cable_fit():
    for i in range(3):
            ind = i+1
            pts, _, _ = data(index=ind, segment=True, data_name='plug').load(pose=True, kp_delta_th=0.005)
            constraint = CableConstraint()
            constraint.fit(pts[1], h_inf=True)
            print(
            f"** Ground truth **\nradius_1:\n: {0.28}\nrest_pt:\n: {np.array([-0.31187662, -0.36479221, -0.03707742])}")

def rake_fit():
    pts, _, _ = data(index=1, segment=True, data_name='rake').load(pose=True, kp_delta_th=0.005)
    constraint = DoublePointConstraint()
    constraint.fit(pts[1], h_inf=True)
            
def set_params():
    params = {'rest_pt': np.array([0,0,0]), 'pt': np.array([0.5,0.0,0.0])}
    const = PointConstraint()
    const.set_params(params)
    test = np.array([0.01,1.01,0.01,0.01,-0.01,0.02])
    print(const.jac_fn(test))

def save_cset():
    names = ['cable_fixture', 'front_pivot']
    constraints = [CableConstraint,
                   CableConstraint]  # list of constraints
    datasets, _, _ = data(index=1, segment=True, data_name='plug').load(pose=True, kp_delta_th=0.005)
    
    c_set = ConstraintSet()
    c_set.fit(names=names, constraints=constraints, datasets=datasets)

    path = os.getcwd() + "/contact_monitoring/data/cable_constraint.pickle"
    c_set.save(file_path=path)
    c_set.load(file_path=path)
    for i in range(2): print(c_set.constraints)

def test_similarity():
    path = os.getcwd() + "/contact_monitoring/data/cable_constraint.pickle"
    c_set = ConstraintSet(file_path = path)
    print(c_set.constraints)
    input_x = np.eye(4)
    input_x[:3,-1] = np.array([0,20,0])
    input_force = np.array([0.2,-4.5,0])
    c_set.id_constraint_force(input_x, input_force)

if __name__ == "__main__":
    print("** Starting test(s) **")
    #cable_fit()
    rake_fit()
    #set_params()
    #save_cset()
    #test_similarity()





