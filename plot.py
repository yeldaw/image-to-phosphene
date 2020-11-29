import mxnet as mx
import numpy as np
from mxnet.gluon import nn
import os
from gluoncv import utils
mx.random.seed(50)
from matplotlib import pyplot as plt


def display(dir, name, data):
    labels = []
    joints = []
    ids = []
    img = mx.image.imread(dir + name)
    for key in data.keys():
        annorect = data[key]['annorect']
        labels.append(
            [
                annorect['x1'],
                annorect['y1'],
                annorect['x2'],
                annorect['y2']
            ]
        )
        for joint_keys in range(len(annorect['annopoints'].keys())):
            joints.append(
                np.array(
                    (
                        annorect['annopoints'][f'{joint_keys}']['x'],
                        annorect['annopoints'][f'{joint_keys}']['y']
                    )
                )
            )
            ids.append(
                annorect['annopoints'][f'{joint_keys}']['is_visible']
            )
    joints = np.reshape(joints, (1, 16, 2))
    confidence = np.array([0.5 for i in range(16)]).reshape((1, 16, 1))
    labels = np.array(labels)
    names = [f'Person {i}' for i in range(len(labels))]
    ax = plot_keypoints(img, joints, confidence, labels, names)
    ax.plot()


def display_coords(img, coords):
    if isinstance(coords, mx.nd.NDArray):
        img = img.asnumpy().transpose(1, 2, 0)
    if isinstance(coords, mx.nd.NDArray):
        coords = coords.asnumpy()
    joints = np.zeros((16, 2))
    for i in range(int(len(coords)/2)):
        joints[i][0] = coords[i * 2]
        joints[i][1] = coords[i * 2 + 1]
    joints = np.reshape(joints, (1, 16, 2))
    ax = plot_pred(img, joints)
    ax.plot()


def plot_keypoints(img, coords, confidence, labels, class_names, keypoint_thresh=0.2):

    import matplotlib.pyplot as plt

    if isinstance(coords, mx.nd.NDArray):
        coords = coords.asnumpy()
    if isinstance(confidence, mx.nd.NDArray):
        confidence = confidence.asnumpy()

    joint_visible = confidence[:, :, 0] > keypoint_thresh
    joint_pairs = [[0, 1], [1, 2], [2, 3], [2, 6],
                   [5, 4], [4, 3], [3, 6],
                   [6, 7], [7, 8], [8, 9],
                   [10, 11], [11, 12], [15, 14],
                   [14, 13], [13, 8], [14, 8]]
    names = [f'Person {i}' for i in range(len(labels))]
    ax = utils.viz.plot_bbox(img, [], class_names = names)

    colormap_index = np.linspace(0, 1, len(joint_pairs))
    for i in range(coords.shape[0]):
        pts = coords[i]
        for cm_ind, jp in zip(colormap_index, joint_pairs):
            if joint_visible[i, jp[0]] and joint_visible[i, jp[1]]:
                ax.plot(pts[jp, 0], pts[jp, 1],
                        linewidth=3.0, alpha=0.7, color=plt.cm.cool(cm_ind))
                ax.scatter(pts[jp, 0], pts[jp, 1], s=20)
    plt.show()
    return ax


def plot_pred(img, coords):
    if isinstance(coords, mx.nd.NDArray):
        coords = coords.asnumpy()
    joint_pairs = [[0, 1], [1, 2], [2, 3], [2, 6],
                   [5, 4], [4, 3], [3, 6],
                   [6, 7], [7, 8], [8, 9],
                   [10, 11], [11, 12], [15, 14],
                   [14, 13], [13, 8], [12, 8]]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(img)
    colormap_index = np.linspace(0, 1, len(joint_pairs))
    for i in range(coords.shape[0]):
        pts = coords[i]
        for cm_ind, jp in zip(colormap_index, joint_pairs):
            ax.plot(pts[jp, 0], pts[jp, 1],
                    linewidth=3.0, alpha=0.7, color=plt.cm.cool(cm_ind))
            ax.scatter(pts[jp, 0], pts[jp, 1], s=20)
    return ax

