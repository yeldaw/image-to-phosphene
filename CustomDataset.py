"""Dataset container."""
__all__ = ['CustomDataset']

import os
import numpy as np
import mxnet.gluon.data.vision.datasets as datasets
from mxnet.gluon.data import dataset
from mxnet import nd, base, ndarray, gpu
from mxnet.gluon import loss as loss_class
import matplotlib.pyplot as plt
from mpii_load import load_json


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


def create_heatmap(x_loc, y_loc, x_size=64, y_size=64, num=1):
    # Initializing value of x-axis and y-axis
    # in the range -1 to 1
    x, y = np.meshgrid(np.linspace(-num, num, x_size), np.linspace(-num, num, y_size))

    x = (x + (x_size/2-x_loc)/(x_size/2) * num)
    y = (y + (y_size/2-y_loc)/(y_size/2) * num)


    # print(-(x_size-x_loc)/x_size)
    # print(-(y_size-y_loc)/y_size)

    dst = np.sqrt(x * x + y * y)
    # Intializing sigma and muu
    sigma = 1
    muu = 0.000

    # Calculating Gaussian array
    gauss = np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2)))
    return gauss


def grab_joints(dic):
    joints = []
    for gen_key in dic.keys():
        annorect = dic[gen_key]['annorect']
        # joints.append(annorect['x1']),
        # joints.append(annorect['x2']),
        # joints.append(annorect['y1']),
        # joints.append(annorect['y2'])
        annopoints = annorect['annopoints']
        for joint in range(len(annopoints)):
            x = annopoints[str(joint)]['x']
            y = annopoints[str(joint)]['y']
            # visible = annopoints[str(joint)]['is_visible']
            # vis = visible if isinstance(visible, int) else 0
            joints.extend((x, y))
    return joints


class CustomDataset(dataset.Dataset):
    """Proper class for custom datasets"""

    def __init__(self, root, data_file, label_file, X, y, label_X, label_y, flag=0, transform=None):
        super(CustomDataset, self).__init__()
        self._transform = transform
        self.dim_x = X
        self.dim_y = y
        self.label_X = label_X
        self.label_y = label_y
        self.x_factor = label_X/self.dim_x
        self.y_factor = label_y/self.dim_y
        self._flag = flag
        self._data = None
        self._keys = None
        self._label = None
        self._new_labels = None
        root = os.path.expanduser(root)
        self._data_path = os.path.join(root, data_file)
        self._label_file = load_json(root, label_file)
        self._get_data()

    def __getitem__(self, idx):
        if self._transform is not None:
            return self._transform(nd.array(self._data[idx], ctx=gpu(0))), nd.array(self._label[idx], ctx=gpu(0))
        return nd.array(self._data[idx], dtype='uint8', ctx=gpu(0)), nd.array(self._label[idx], ctx=gpu(0))

    def __len__(self):
        return len(self._keys)

    def _get_data(self):
        self._keys = os.listdir(self._data_path)[0:1]
        labels = []
        data_file = []
        for key in self._keys:
            data_file.append(plt.imread(self._data_path + key, self._flag))
            labels.append(grab_joints(self._label_file[key]))
        self._data = np.array(data_file)
        heatmap_labels = []
        for label in labels:
            points = []
            for i, j in zip(label[0::2], label[1::2]):
                points.append(create_heatmap(i * self.x_factor, j * self.y_factor, self.label_X, self.label_y, num=2))
            heatmap_labels.append(points)
        self._label = heatmap_labels

