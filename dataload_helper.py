import pickle
import os
import numpy as np
from rotation_helpers import xyz_to_rotation

class data():

    def __init__(self, clustered):
        self.directory = os.path.dirname(os.path.realpath(__file__)) + "/data/"
        if clustered:
            self.directory += "clustering_video_"
        else:
            self.directory += "video_"
    def load_data(self, path):
        return pickle.load(open(path, "rb"))

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

    def load(self):
        if os.path.isfile(self.path):
            dataset = np.array(self.load_data(self.path))
            print(f"\nLoaded {len(dataset)} samples from plug dataset {self.experiment}")
            #generate list of segments
            if self.segment:
                dataset_segments = []
                for segment in range(int(max(dataset[:, -1])) + 1):
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
    """
    def video(self):
        data.__init__(self, clustered=False)
        self.directory += "plug_"
        self.directory += self.experiment
        self.path = self.directory + str(self.index) + ".mp4"
        self.play_video(self.path)
    """

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
    def points(self):
        pt = np.array([[0, 0, 0.5], [0, 0.01, 0.52], [0.02, -0.01, 0.49]])
        pt = np.transpose(pt[:self.n_points])
        print("Debug")
        print(pt)

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
    def __init__(self, index=1, clustered=False, segment=True):
        self.segment=segment
        self.index = index
        data.__init__(self, clustered=clustered)
        self.directory += "rake_"
        self.path = self.directory + str(self.index) + ".pickle"

    def load(self, center=False):
        if os.path.isfile(self.path):
            dataset = np.array(self.load_data(self.path))
            print(f"\nLoaded {len(dataset)} samples from rake dataset")
            if center:
                dataset = np.mean(dataset, axis=1)[:, None, :]

            #generate list of segments
            if self.segment:
                dataset_segments = []
                for segment in range(int(max(dataset[:, 0, -1])) + 1):
                    dataset_segments.append(dataset[dataset[:, 0, -1] == segment, :, :3][:,0,:])
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



if __name__ == "__main__":
    pts = point_c_data(n_points=1).points()
    print(np.array(pts).shape)


