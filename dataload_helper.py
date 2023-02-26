import pickle
import os
import numpy as np
from .rotation_helpers import xyz_to_rotation
from .kp2pose import *

class data():

    def __init__(self, index=1, clustered=False, segment=False, data_name='rake'):
        self.directory = os.path.dirname(os.path.realpath(__file__)) + "/data/"
        if clustered:
            self.directory += "clustering_video_"
        else:
            self.directory += "video_"
        self.directory += data_name + "_"
        self.path = self.directory + str(index) + ".pickle"
        self.segment = segment
        self.index = index
        self.clustered = clustered

    def load(self, pose=True, kp_delta_th=0.005):
        if os.path.isfile(self.path):
            dataset = np.array(pickle.load(open(self.path, "rb")))
            n_dataset = len(dataset)
            print(f"\nLoaded {n_dataset} samples from {self.path} dataset {self.index}")

            dataset_len = 0

            if dataset.ndim > 2:
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
                if _dataset.shape[0] > 0.5 * dataset.shape[0]:
                    dataset = _dataset
                    dataset_segments = _dataset_segments
                    dataset_time = _dataset_time
                    print(
                        f"Discarded {n_discard_opti} poses due to bad correspondence of kp_data, {n_dataset - n_discard_opti} poses remaining")
                else:
                    raise Exception(
                        "had to discard more than 50% of poses, there is something wrong with the dataset")
            else:
                dataset_segments = dataset[:,-1]
                dataset_time = dataset[:,-2]
                if pose:

                    dataset_T = np.repeat(np.eye(4)[None,:,:], dataset.shape[0], axis = 0)
                    dataset_T[:, :3, 3] = dataset[:, :3]
                    dataset = dataset_T



            #generate list of segments
            if self.segment:
                dataset_segmented = []
                for segment in range(int(max(dataset_segments) + 1)):
                    if dataset.ndim > 2:
                        if pose: dataset_segmented.append(dataset[dataset_segments == segment, :, :])
                        else: dataset_segmented.append(dataset[dataset_segments == segment, :, :3])
                    else:
                        if pose: dataset_segmented.append(dataset[dataset_segments == segment, :, :3])
                        else: dataset_segmented.append(dataset[dataset[:, -1] == segment, :3])
                print(
                    f"Dataset contains {len(dataset_segmented)} segments:")
                try:
                    print(f"Segment 1: {len(dataset_segmented[0])} samples")
                    print(f"Segment 2: {len(dataset_segmented[1])} samples")
                    print(f"Segment 3: {len(dataset_segmented[2])} samples")
                except:
                    pass

                dataset = dataset_segmented

            if False:#self.clustered:  # Throw away first and last 10 percent of clusters
                dataset_segmented_10p = []
                for data_s in dataset_segmented:
                    index_10p = int(data_s.shape[0]/10)
                    dataset_segmented_10p.append(data_s[index_10p:-index_10p])
                print("Cut of data: Reduced size from Original segments of shape")
                for data_s in dataset_segmented:
                    print(f"shape: {data_s.shape}")
                print("To new shapes:")
                for data_s in dataset_segmented_10p:
                    print(f"shape: {data_s.shape}")
                dataset = dataset_segmented_10p
            return dataset, dataset_segments, dataset_time

        else:
            print(f"No data at {self.path}")


if __name__ == "__main__":
    # data_name = rake / plug
    # segment = True -> [Dataset_segment_0, Dataset_segment_1, Dataset_segment_2, ...]
    # segment = False -> stack([Dataset_segment_0, Dataset_segment_1, Dataset_segment_2, ...])
    # pose = True -> Dataset[i,:,:] = (4,4) gives poses for every datapoint
    # pose = False -> Dataset[i,:,:] = (i,3) gives keypoints for every datapoint
    for i in range(3):
        dataset, segments, time = data(index=i+1, segment=True, data_name="plug").load(pose=True, kp_delta_th=0.005)
        print(dataset[0][0,:,:])

