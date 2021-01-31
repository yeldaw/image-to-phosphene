"""Dataset container."""
__all__ = ['CustomDataset']

import os, random
import numpy as np
import mxnet.gluon.data.vision.datasets as datasets
from mxnet.gluon.data import dataset
from mxnet import nd, base, ndarray, gpu
from mxnet.gluon import loss as loss_class
import matplotlib.pyplot as plt
from mpii_load import load_json
from scipy.ndimage import gaussian_filter


def _apply_weighting(F, loss, weight=None, sample_weight=None):
    """Apply weighting to loss.

    Parameters
    ----------
    loss : Symbol
        The loss to be weighted.
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch separately, `sample_weight` should have
        shape (64, 1).

    Returns
    -------
    loss : Symbol
        Weighted loss
    """
    if sample_weight is not None:
        loss = F.broadcast_mul(loss, sample_weight)

    if weight is not None:
        assert isinstance(weight, base.numeric_types), "weight must be a number"
        loss = loss * weight

    return loss


def _reshape_like(F, x, y):
    """Reshapes x to the same shape as y."""
    return x.reshape(y.shape) if F is ndarray else F.reshape_like(x, y)


def create_dataset(directory, imglist, flag=0):
    return datasets.ImageListDataset(directory, imglist, flag)


def load_dataset(batch_size=10):
    dataset = CustomDataset('F:\\Thesis Datasets\\test', 'data.npz', 'label.npz', 5, 5)
    # transformed_dataset = dataset.transform_first(transformer)
    # traindata = mx.gluon.data.DataLoader(transformed_dataset, batch_size=batch_size)
    # return traindata


# def create_heatmap(x_loc, y_loc, x_size=64, y_size=64, num=1):
#     # Initializing value of x-axis and y-axis
#     # in the range -1 to 1
#     x, y = np.meshgrid(np.linspace(-num, num, x_size), np.linspace(-num, num, y_size))
#
#     x = (x + (x_size/2-x_loc)/(x_size/2) * num)
#     y = (y + (y_size/2-y_loc)/(y_size/2) * num)
#
#
#     # print(-(x_size-x_loc)/x_size)
#     # print(-(y_size-y_loc)/y_size)
#
#     dst = np.sqrt(x * x + y * y)
#     # Intializing sigma and muu
#     sigma = 1
#     muu = 0.000
#
#     # Calculating Gaussian array
#     gauss = np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2)))
#     return gauss


# def create_heatmap(x_loc, y_loc, x_size=64, y_size=64, intensity=1, sigma=2):
#     heatmap = np.zeros((x_size, y_size))
#     heatmap[int(y_loc), int(x_loc)] = intensity  # EDIT: Klopt dit wel, qua x en y coordinaten? Meestal is andersom toch?
#     return gaussian_filter(heatmap, sigma=sigma)


def create_heatmap(x_loc, y_loc, x_size=64, y_size=64, num=5):
    heatmap = np.zeros((x_size, y_size))
    chance = 1 / (num ** 2)
    for i in range(-num, num + 1):
        for j in range(-num, num + 1):
            if x_size > int(x_loc + j) >= 0 and y_size > int(y_loc + i) >= 0:
                heatmap[int(y_loc + i)][int(x_loc + j)] = (num - abs(i)) * (num - abs(j)) * chance * 1000
    return heatmap


def grab_joints(dic):
    joints = []
    visibility = []
    for gen_key in dic.keys():
        annorect = dic[gen_key]['annorect']
        annopoints = annorect['annopoints']
        for joint in range(len(annopoints)):
            x = annopoints[str(joint)]['x']
            y = annopoints[str(joint)]['y']
            visible = annopoints[str(joint)]['is_visible']
            vis = visible if isinstance(visible, int) else 0
            joints.extend((x, y))
            visibility.append(vis)
    return joints, visibility


def grab_triplet(dic):
    joint_ids = {
         '0': 'medial_right_ankle',
         '1': 'lateral_right_ankle',
         '2': 'medial_right_knee',
         '3': 'lateral_right_knee',
         '4': 'medial_right_hip',
         '5': 'lateral_right_hip',
         '6': 'medial_left_hip',
         '7': 'lateral_left_hip',
         '8': 'medial_left_knee',
         '9': 'lateral_left_knee',
         '10': 'medial_left_ankle',
         '11': 'lateral_left_ankle',
         '12': 'right_neck',
         '13': 'left_neck',
         '14': 'medial_right_wrist',
         '15': 'lateral_right_wrist',
         '16': 'medial_right_bow',
         '17': 'lateral_right_bow',
         '18': 'medial_right_shoulder',
         '19': 'lateral_right_shoulder',
         '20': 'medial_left_shoulder',
         '21': 'lateral_left_shoulder',
         '22': 'medial_left_bow',
         '23': 'lateral_left_bow',
         '24': 'medial_left_wrist',
         '25': 'lateral_left_wrist',
    }
    joints = []
    for key in joint_ids:
        id = joint_ids[key]
        x = dic[id]['x']
        y = dic[id]['y']
        joints.extend((x, y))
    return joints


def make_heatmap(list_of_data, x, y, size=64):
    heatmaps = []
    for i, j in zip(list_of_data[0::2], list_of_data[1::2]):
        heatmaps.append(create_heatmap(i * x, j * y, x_size=size, y_size=size))
    return heatmaps


class CustomDataset(dataset.Dataset):
    """Proper class for custom datasets"""

    def __init__(self, root, data_file, label_file, X, y, label_X, label_y, flag=0, transform=None, multiplier=500,
                 triplet_file=None):
        super(CustomDataset, self).__init__()
        self._transform = transform
        self.dim_x = X
        self.dim_y = y
        self.label_X = label_X
        self.label_y = label_y
        self.x_factor = label_X / self.dim_x
        self.y_factor = label_y / self.dim_y
        self._flag = flag
        self._data = None
        self._keys = None
        self._label = None
        self._triplets = None
        self._visibility = None
        self._new_labels = None
        self._data_path = os.path.join(root, data_file)
        self._label_file = os.path.join(root, label_file)
        self.counter = 0
        self.multiplier = multiplier
        self._triplet_dir = None
        if triplet_file is not None:
            self._triplet_dir = os.path.join(root, triplet_file)
            self.get_triplet()
        else:
            self.get_data()

    def __getitem__(self, idx):
        if self._transform is not None:
            if self._triplet_dir is not None:
                labels = make_heatmap(self._label[idx], 1, 1, 256)
                triplets = make_heatmap(self._triplets[idx], self.x_factor, self.y_factor)
                return self._transform(nd.array(self._data[idx], ctx=gpu(0))), nd.array(labels, ctx=gpu(0)), \
                       nd.array(self._visibility[idx], ctx=gpu(0)).reshape(16, 1, 1), \
                       nd.array(triplets, ctx=gpu(0))
            return self._transform(nd.array(self._data[idx], ctx=gpu(0))), \
                   nd.array(make_heatmap(self._label[idx], self.x_factor, self.y_factor), ctx=gpu(0)), \
                nd.array(self._visibility[idx], ctx=gpu(0)).reshape(16, 1, 1)
        return nd.array(self._data[idx], dtype='uint8', ctx=gpu(0)), nd.array(self._label[idx], ctx=gpu(0))

    def __len__(self):
        return len(self._keys)

    def get_data(self, update=False):
        self._keys = random.sample(os.listdir(self._data_path), self.multiplier)
        #[self.counter * self.multiplier:(self.counter + 1) * self.multiplier]
        labels = [[]] * self.multiplier
        visibility = [[]] * self.multiplier
        data_file = [None] * self.multiplier
        for file in os.listdir(self._label_file):
            label = load_json(self._label_file, file)
            for index in range(len(self._keys)):
                key = self._keys[index]
                if key in label.keys():
                    data_file[index] = plt.imread(self._data_path + key, self._flag)
                    joints, visible = grab_joints(label[key])
                    labels[index] = joints
                    visibility[index] = visible
        self._data = np.array(data_file)
        # heatmap_labels = []
        # for label in labels:
        #     points = []
        #     for i, j in zip(label[0::2], label[1::2]):
        #         points.append(create_heatmap(i * self.x_factor, j * self.y_factor, self.label_X, self.label_y))
        #     heatmap_labels.append(points)
        # self._label = heatmap_labels
        self._label = labels
        self._visibility = visibility
        # if (self.counter + 1) * self.multiplier > len(os.listdir(self._data_path)):
        #     self.counter = 0
        # else:
        #     self.counter += 1

    def get_triplet(self, update=False):
        self._keys = random.sample(os.listdir(self._data_path), self.multiplier)
        labels = [[]] * self.multiplier
        visibility = [[]] * self.multiplier
        data_file = [None] * self.multiplier
        triplets = [[]] * self.multiplier
        for file in os.listdir(self._triplet_dir):
            triplet = load_json(self._triplet_dir, file)
            label = load_json(self._label_file, "Joint" + file.split("Triplet")[1])
            for index in range(len(self._keys)):
                key = self._keys[index]
                if key in triplet.keys():
                    data_file[index] = plt.imread(self._data_path + key, self._flag)
                    joints, visible = grab_joints(label[key])
                    labels[index] = joints
                    triplets[index] = grab_triplet(triplet[key])
                    visibility[index] = visible
        self._data = np.array(data_file)
        # heatmap_labels = []
        # for label in labels:
        #     points = []
        #     for i, j in zip(label[0::2], label[1::2]):
        #         points.append(create_heatmap(i * self.x_factor, j * self.y_factor,
        #                                      64 if not self._triplet_file else 256,
        #                                      64 if not self._triplet_file else 256))
        #     heatmap_labels.append(points)
        # self._label = heatmap_labels
        self._label = labels
        self._triplets = triplets
        self._visibility = visibility
        # if (self.counter + 1) * self.multiplier > len(os.listdir(self._data_path)):
        #     self.counter = 0
        # else:
        #     self.counter += 1