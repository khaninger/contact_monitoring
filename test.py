import os
from .dataload_helper import data
from .visualize import plot_x_pt_inX, plot_3d_points_segments, plot_T_traj, plot_T_traj_premium
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



def cable_fit_all():
    data_set = []
    for i in range(3):
        ind = i+1
        pts, _, _ = data(index=ind, segment=True, data_name='plug').load(pose=True, kp_delta_th=0.005)
        pts_thread, _, _ = data(index=ind, segment=True, data_name='plug_threading').load(pose=True, kp_delta_th=0.005)
        data_set.append(pts[1])
        data_set.append(pts_thread[1])
    data_set = np.vstack(data_set)
    constraint = CableConstraint()
    constraint.fit(data_set, h_inf=True)
    print(
    f"** Ground truth **\nradius_1:\n: {0.28}\nrest_pt:\n: {np.array([-0.31187662, -0.36479221, -0.03707742])}")

    data_set = []
    for i in range(3):
        ind = i + 1
        pts_thread, _, _ = data(index=ind, segment=True, data_name='plug_threading').load(pose=True, kp_delta_th=0.005)
        data_set.append(pts_thread[2])
    data_set = np.vstack(data_set)
    constraint = CableConstraint()
    constraint.fit(data_set, h_inf=True)
    print(
        f"** Ground truth **\nradius_1:\n: {0.14}\nrest_pt:\n: {np.array([-0.22657112, -0.47936426, -0.0320753])}")

def rake_fit():
    pts, _, _ = data(index=1, segment=True, data_name='rake').load(pose=True, kp_delta_th=0.005)
    constraint = RakeConstraint_2pt()
    constraint.fit(pts[1], h_inf=True)

def set_params():
    params = {'rest_pt': np.array([0,0,0]), 'pt': np.array([0.5,0.0,0.0])}
    const = PointConstraint()
    const.set_params(params)
    test = np.array([0.01,1.01,0.01,0.01,-0.01,0.02])
    print(const.jac_fn(test))

def load_plug_data(L=[1,2,3], clustered=False):
    cable_fixture = []
    free_space = []
    front_pivot = []
    for ind in L:
        pts, _, _ = data(index=ind, clustered = clustered, segment=True, data_name='plug').load(pose=True, kp_delta_th=0.005)
        pts_thread, _, _ = data(index=ind, clustered= clustered, segment=True, data_name='plug_threading').load(pose=True, kp_delta_th=0.005)
        free_space.append(pts[0])
        free_space.append(pts_thread[0])
        cable_fixture.append(pts[1])
        cable_fixture.append(pts_thread[1])
        front_pivot.append(pts_thread[2])
    front_pivot = np.vstack(front_pivot)
    cable_fixture = np.vstack(cable_fixture)
    free_space = np.vstack(free_space)
    return front_pivot, cable_fixture, free_space


def load_rake_data(L=[1,2,3], clustered=False, rakedata = 'sexy_rake'):
    rake_free = []
    rake_mode = []
    for ind in L:
        pts, _, _ = data(index=ind, clustered = clustered, segment=True, data_name=rakedata).load(pose=True, kp_delta_th=0.005)
        if rakedata == 'sexy_rake_hinge':
            rake_free.append(pts[0])
            rake_mode.append(pts[2])
        else:
            rake_free.append(pts[0])
            rake_mode.append(pts[1])

    rake_free = np.vstack(rake_free)
    rake_mode = np.vstack(rake_mode)

    return rake_free, rake_mode

def plot_rake_hinge_fit(L=[1], clustered=False):
    rake_free, rake_hinge = load_rake_data(L=L, clustered=clustered, rakedata='sexy_rake_hinge')
    const = RakeConstraint_Hinge()
    params = const.fit(data=rake_hinge, h_inf=False)
    plot_T_traj_premium(rake_hinge, points=[params['pt_0'], params['pt_1']])

def subsample(data, sub=10):
    # Subsample
    data_subset = []
    for i, T in enumerate(data):
        if i % sub == 0:
            data_subset.append(T)
    return np.array(data_subset)

def plot_rake_fit(L=[1, 2, 3], clustered=False):
    rake_free, rake_plane = load_rake_data(L=L, clustered=clustered)
    const = RakeConstraint_2pt()
    params = const.fit(rake_plane, h_inf=True)
    plane = [params['plane_normal'], params['d']]
    points = [params['pt_0'],params['pt_1']]
    rake_plane = subsample(rake_plane)
    plot_T_traj_premium(rake_plane, plane=plane, points=points, plane_gt=[[0,0,1], 0.04])

def plot_plug_fit(L=[1,2,3], clustered=True):
    front_pivot, cable_fixture, free_space = load_plug_data(L, clustered=clustered)
    data = [cable_fixture, front_pivot, free_space]

    #const = CableConstraint()
    #params = const.fit(cable_fixture, h_inf=True)
    #plot_3d_points_segments(data, rest_pt=params['rest_pt'], rest_pt_gt=np.array([-0.31187662, -0.36479221, -0.02907742]), radius=params['radius_1'], exp_n=0)

    const = CableConstraint()
    params = const.fit(front_pivot, h_inf=True)
    plot_3d_points_segments(data, rest_pt=params['rest_pt'],
                            rest_pt_gt=np.array([-0.22657112, -0.47936426, -0.0290753]), radius=params['radius_1'],
                            exp_n=0)


def save_cset_cable():
    #load all cable data and concatenate
    cable_fixture = []
    free_space = []
    front_pivot = []
    for i in range(3):
        ind = i + 1
        pts, _, _ = data(index=ind, segment=True, data_name='plug').load(pose=True, kp_delta_th=0.005)
        pts_thread, _, _ = data(index=ind, segment=True, data_name='plug_threading').load(pose=True, kp_delta_th=0.005)
        free_space.append(pts[0])
        free_space.append(pts_thread[0])
        cable_fixture.append(pts[1])
        cable_fixture.append(pts_thread[1])
        front_pivot.append(pts_thread[2])


    front_pivot = np.vstack(front_pivot)
    cable_fixture = np.vstack(cable_fixture)
    free_space = np.vstack(free_space)

    names = ['free_space', 'cable_fixture', 'front_pivot']

    constraints = [FreeSpace,
                   CableConstraint,
                   CableConstraint]

    datasets = [free_space, cable_fixture, front_pivot]

    c_set = ConstraintSet()
    c_set.fit(names=names, constraints=constraints, datasets=datasets)

    path = os.getcwd() + "/contact_monitoring/data/cable_constraint.pickle"
    c_set.save(file_path=path)
    c_set.load(file_path=path)
    print(c_set.constraints)



def save_cset_rake():
    #load rake data
    dataset, segments, time = data(index=1, segment=True, data_name="sexy_rake").load(pose=True, kp_delta_th=0.003)
    rake_on_plane = dataset[1]
    rake_freespace = dataset[0]
    dataset, segments, time = data(index=1, segment=True, data_name="sexy_rake_hinge").load(pose=True, kp_delta_th=0.003)
    rake_on_hinge = dataset[2]

    names = ['free_space', 'plane', 'hinge']

    constraints = [FreeSpace,
                   RakeConstraint_2pt,
                   RakeConstraint_Hinge]

    datasets = [rake_freespace, rake_on_plane, rake_on_hinge]

    c_set = ConstraintSet()
    c_set.fit(names=names, constraints=constraints, datasets=datasets)

    path = os.getcwd() + "/contact_monitoring/data/rake_constraint.pickle"
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
        constraint = RakeConstraint_1pt()
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
        constraint = RakeConstraint_2pt()
        params = constraint.fit(data=rake_on_plane, h_inf=True)
        plane = [params['plane_normal'], params['d']]
        points = [params['pt_0'], params['pt_1']]
        plot_x_pt_inX(L_pt=points, X=rake_on_plane, plane=plane)
        # print plane z height at xy mean of data
        xy_mean = np.mean(rake_on_plane[:,:2,3], axis=0)
        z_height = np.dot(params['plane_normal'],np.append(xy_mean,0))
        print(f"\n****RESULTS***\n\nz_height of plane at xy mean of data:\n{z_height}")
        print(f"\n\nPlane Normal at:\n{params['plane_normal']}")



def test_rake_fit_hinge():
    for i in range(3):
        dataset, segments, time = data(index=i+1, segment=True, data_name="sexy_rake_hinge").load(pose=True, kp_delta_th=0.003)
        rake_on_hinge = dataset[2]

        constraint = RakeConstraint_Hinge()
        params = constraint.fit(data=rake_on_hinge, h_inf=False)
        points = [params['pt_0'], params['pt_1']]
        plot_x_pt_inX(L_pt=points, X=rake_on_hinge, plane=None)


if __name__ == "__main__":
    print("** Starting test(s) **")
    #cable_fit()
    #cable_fit_all()
    #save_cset_cable()
    #test_rake_fit_one_pt()
    #test_rake_fit_two_pt()
    #set_params()
    #save_cset()
    #test_similarity()
    #save_cset_rake()
    #test_rake_fit_hinge()

    #Plot Figures
    #plot_plug_fit(L=[2, 3], clustered=True)
    plot_rake_fit(L=[1], clustered=False)
    #plot_rake_hinge_fit(L=[2], clustered=False)
