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
    # plug_threading
    # load threading data
    dataset, segments, time = data(index=2, segment=True, data_name="plug_threading").load(pose=True, kp_delta_th=0.005)
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

def test_similarity():
    path = os.getcwd() + "/contact_monitoring/data/cable_constraint.pickle"
    c_set = ConstraintSet(file_path = path)
    print(c_set.constraints)
    input_x = np.eye(4)
    input_x[:3,-1] = np.array([0,20,0])
    input_force = np.array([0.2,-4.5,0])
    c_set.id_constraint_force(input_x, input_force)

def test_rake_fit_one_pt():
    for i in range(3):
        dataset, segments, time = data(index=i+1, segment=True, data_name="sexy_rake").load(pose=True, kp_delta_th=0.003)
        rake_on_plane = dataset[1]
        constraint = RakeConstraint()
        params = constraint.fit(data=rake_on_plane, h_inf=True)
        points = [params['pt']]
        plane = [params['plane_normal'], params['d']]
        plot_x_pt_inX(L_pt=points, X=rake_on_plane, plane=plane)
        # print plane z height at xy mean of data
        xy_mean = np.mean(rake_on_plane[:,:2,3], axis=0)
        z_height = np.dot(params['plane_normal'],np.append(xy_mean,0))
        print(f"\n****RESULTS***\n\nz_height of plane at xy mean of data:\n{z_height}")
        print(f"\n\nPlane Normal at:\n{params['plane_normal']}")


def test_rake_fit_two_pt():
    for i in range(3):
        dataset, segments, time = data(index=i+1, segment=True, data_name="sexy_rake").load(pose=True, kp_delta_th=0.003)
        rake_on_plane = dataset[1]
        constraint = RakeConstraint3()
        params = constraint.fit(data=rake_on_plane, h_inf=True)
        plane = [params['plane_normal'], params['d']]
        points = [params['pt_0'], params['pt_1']]
        plot_x_pt_inX(L_pt=points, X=rake_on_plane, plane=plane)
        # print plane z height at xy mean of data
        xy_mean = np.mean(rake_on_plane[:,:2,3], axis=0)
        z_height = np.dot(params['plane_normal'],np.append(xy_mean,0))
        print(f"\n****RESULTS***\n\nz_height of plane at xy mean of data:\n{z_height}")
        print(f"\n\nPlane Normal at:\n{params['plane_normal']}")

if __name__ == "__main__":
    print("** Starting test(s) **")
    #cable_fit()
    #test_rake_fit_one_pt()
    test_rake_fit_two_pt()
    #set_params()
    #save_cset()
    #test_similarity()





