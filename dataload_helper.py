import pickle
import os
import numpy as np
from .rotation_helpers import xyz_to_rotation
from .kp2pose import *

class data():

    def __init__(self, clustered=False):
        self.directory = os.path.dirname(os.path.realpath(__file__)) + "/data/"
        if clustered:
            self.directory += "clustering_video_"
        else:
            self.directory += "video_"

    def load_data(self, path):
        return pickle.load(open(path, "rb"))

    def save_data(self, data, constraint="cable", specifier='0'):
        path = os.path.dirname(os.path.realpath(__file__)) + "/data/constraint_fit_" + constraint + ".pickle"
        try:
            dictionary = self.load_data(path)
        except:
            dictionary = {}

        dictionary[specifier] = data
        pickle.dump(dictionary, open(path, "wb"))
        print(f"Saved {constraint} data at {path}")
        print(dictionary)

    """
    def play_video(self, path):
        print(path)
        os.system("start " + path)
    """
class plug(data):
    def __init__(self, experiment="", index=1, clustered=False, segment=True):
        # experiments are
        # ""
        # "less_"
        # "threading_"
        self.segment=segment
        self.experiment = experiment
        self.index = index
        data.__init__(self, clustered=clustered)
        self.directory += "plug_"
        self.directory += self.experiment
        self.path = self.directory + str(self.index) + ".pickle"

    def save(self, rest_pt=None, radius=None, overwrite=False, specifier='0', T=None, index=0):
        if overwrite:
            print("Delete old data")
            dictionary = {}
        else:
            try:
                path = os.getcwd() + "/data/" + "plug_constraint_" + str(index) + ".pickle"
                dictionary = self.load_data(path=path)
            except:
                dictionary = {}

        if specifier == 'T':
            specifier_data = T
        else:
            specifier_data = {'rest_pt': rest_pt, 'radius': radius}
        dictionary[specifier] = specifier_data
        self.save_data(dictionary, path)
        print(f"Saved dict ({dictionary}) for plug_constraint_{index} at {path}")


    def load(self):
        if os.path.isfile(self.path):
            dataset = np.array(self.load_data(self.path))
            print(f"\nLoaded {len(dataset)} samples from plug dataset {self.experiment}")
            #generate list of segments
            if self.segment:
                dataset_segments = []
                for segment in range(int(max(dataset[:, -1])) + 1):
                    T = np.eye(4)
                    T[:3,-1] = dataset[dataset[:, -1] == segment, :3]
                    print(T)
                    dataset_segments.append(dataset[dataset[:, -1] == segment, :3])
                print(
                    f"Dataset contains {len(dataset_segments)} segments:")
                try:
                    print(f"Segment 1: {len(dataset_segments[0])} samples")
                    print(f"Segment 2: {len(dataset_segments[1])} samples")
                    print(f"Segment 3: {len(dataset_segments[2])} samples")
                except:
                    pass

                dataset = dataset_segments
            return dataset

    def random(self, rest_pt = np.array([0.1, 0.2, 0.3]), pt = np.array([0., 0., 0.5]), noise=0.01, rot_0=.4, rot_1=.2):

        Tmats = []
        std = noise
        mean = 0
        noise_t = np.random.normal(mean, std, size=(3,))
        noise_r = np.random.normal(mean, std, size=(3,))
        for x_rot in np.linspace(-rot_0/2, rot_0/2, 100):
            for y_rot in np.linspace(-rot_1/2, rot_1/2, 100):
                rotation = np.array([x_rot, y_rot, 0])+noise_r
                new_rotmat = xyz_to_rotation(rotation)
                new_pt = new_rotmat @ pt + rest_pt
                Tmat = np.eye(4)
                Tmat[:3, 3] = np.squeeze(new_pt) + noise_t
                Tmat[:3, :3] = new_rotmat
                Tmats.append(Tmat)
        Tmats=np.array(Tmats)
        index = np.random.choice(len(Tmats), 25, replace=False).astype(int)

        return Tmats[index]

class point_c_data:

    def __init__(self, n_points=1, rest_pt = np.array([0.1, 0.2, 0.3]), noise=0.01, rot_0=.4, rot_1=.2):
        self.rot_0 = rot_0
        self.rot_1 = rot_1
        self.n_points = n_points
        self.rest_pt = rest_pt
        self.noise_t = np.random.normal(0, noise, size=(3,n_points))
        self.noise_r = np.random.normal(0, noise, size=(3,))
    def points(self, radius=.5):
        pt = np.array([[0, 0, radius], [0, 0.01, 0.52], [0.02, -0.01, 0.49]])
        pt = np.transpose(pt[:self.n_points])
        t_pts = []
        for x_rot in np.linspace(-self.rot_0/2, self.rot_0/2, 100):
            for y_rot in np.linspace(-self.rot_1/2, self.rot_1/2, 100):
                rotation = np.array([x_rot, y_rot, 0])+self.noise_r
                new_rotmat = xyz_to_rotation(rotation)
                new_pt = np.transpose(np.asarray(new_rotmat @ pt)) + np.tile(self.rest_pt,(self.n_points,1))
                t_pts.append(new_pt)
        t_pts = np.array(t_pts)
        index = np.random.choice(len(t_pts), 100, replace=False).astype(int)
        return t_pts[index]

class rake(data):
    def __init__(self, index=1, clustered=False, segment=False, specifier="rake_"):
        self.segment=segment
        self.index = index
        data.__init__(self, clustered=clustered)
        self.directory += specifier
        self.path = self.directory + str(self.index) + ".pickle"

    def load(self, center=False, pose=False, kp_delta_th=0.005):
        if os.path.isfile(self.path):
            dataset = np.array(self.load_data(self.path))
            n_dataset = len(dataset)
            print(f"\nLoaded {n_dataset} samples from rake dataset {self.index}")

            dataset_len = 0
            for init_kp_index in range(int(.1*n_dataset)):
                n_discard = 0
                kp_dim = 3
                init_kp = dataset[init_kp_index,:,:kp_dim]
                T_init = init_T(init_kp)
                __dataset = []
                __dataset_segments = []
                __dataset_time = []
                for i in range(dataset.shape[0]):
                    data_kp = dataset[i,:,:kp_dim]
                    T_rake, delta = find_T(init_kp, data_kp, kp_delta_th=kp_delta_th)
                    if not delta: # only append poses wher kp2kp transform results in per keypoint offset smaller than kp_delta_th
                        if pose:
                            __dataset.append(T_rake @ T_init)
                        else:
                            __dataset.append(dataset[i, :, :])
                        __dataset_segments.append(dataset[i, 0, -1])
                        __dataset_time.append(dataset[i, 0, -2])
                    else:
                        n_discard += 1
                if len(__dataset) > dataset_len:
                    dataset_len = len(__dataset)
                    n_discard_opti = np.copy(n_discard)
                    _dataset_segments = np.array(__dataset_segments)
                    _dataset_time = np.array(__dataset_time)
                    _dataset = np.array(__dataset)

            if _dataset.shape[0] > 0.5*dataset.shape[0]:
                dataset = _dataset
                dataset_segments = _dataset_segments
                dataset_time = _dataset_time
                print(
                    f"Discarded {n_discard_opti} poses due to bad correspondence of kp_data, {n_dataset - n_discard_opti} poses remaining")
            else:
                raise Exception("had to discard more than 50% of poses, there is something wrong with the dataset")


            #generate list of segments
            if self.segment:
                dataset_segmented = []
                if pose:
                    for segment in range(int(max(dataset_segments) + 1)):
                        dataset_segmented.append(dataset[dataset_segments == segment, :, :])
                else:
                    for segment in range(int(max(dataset_segments) + 1)):
                        dataset_segmented.append(dataset[dataset_segments == segment, :, :3])

                print(
                    f"Dataset contains {len(dataset_segmented)} segments:")
                try:
                    print(f"Segment 1: {len(dataset_segmented[0])} samples")
                    print(f"Segment 2: {len(dataset_segmented[1])} samples")
                    print(f"Segment 3: {len(dataset_segmented[2])} samples")
                except:
                    pass

                dataset = dataset_segmented

            if center and not pose and not self.segment:
                dataset = np.mean(dataset, axis=1)


            return dataset, dataset_segments, dataset_time


def max_dist(data):
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

    return np.linalg.norm(hullpoints[bestpair[0]] - hullpoints[bestpair[1]])



if __name__ == "__main__":
    from visualize import plot_x_pt_inX as pp
    #pts = point_c_data(n_points=1).points()
    #print(np.array(pts).shape)
    #for i in range(5):
    #   dataset, segments, time = rake(index=i+1, segment=True).load(center=False, pose=False, kp_delta_th=0.005)

    dataset, segments, time = rake(index=1, segment=True, specifier="sexy_rake_hinge_").load(center=False, pose=False, kp_delta_th=0.005)

    pp(L_pt=dataset[2])
