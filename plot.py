from matplotlib import pyplot as plt
import mxnet as mx
import numpy as np
from gluoncv import utils
import numpy as np

mx.random.seed(50)


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
    confidence = np.array([0.5 for _ in range(16)]).reshape((1, 16, 1))
    labels = np.array(labels)
    names = [f'Person {i}' for i in range(len(labels))]
    ax = plot_keypoints(img, joints, confidence, labels, names)
    ax.plot()


def display_coords(img, coords, triplet=None):
    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy().transpose(1, 2, 0)
    if isinstance(coords, mx.nd.NDArray):
        coords = coords.asnumpy()
    triplet_joints = None
    if triplet is not None:
        if isinstance(triplet, mx.nd.NDArray):
            coords = coords.asnumpy()
        triplet_joints = np.zeros((int(len(triplet) / 2), 2))
        for i in range(int(len(triplet) / 2)):
            triplet_joints[i][0] = triplet[i * 2]
            triplet_joints[i][1] = triplet[i * 2 + 1]
        triplet_joints = np.reshape(triplet_joints, (1, int(len(triplet) / 2), 2))
    joints = np.zeros((int(len(coords)/2), 2))
    for i in range(int(len(coords)/2)):
        joints[i][0] = coords[i * 2]
        joints[i][1] = coords[i * 2 + 1]
    joints = np.reshape(joints, (1, int(len(coords)/2), 2))
    ax = plot_pred(img, joints, triplet_joints)
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
    ax = utils.viz.plot_bbox(img, [], class_names=names)

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


def plot_pred(img, coords, triplets=None):
    if isinstance(coords, mx.nd.NDArray):
        coords = coords.asnumpy()
    triplet_combo = []
    if triplets is not None:
        if isinstance(triplets, mx.nd.NDArray):
            coords = triplets.asnumpy()
        triplet_combo = [[0, 0], [0, 1], [1, 2], [1, 3],
                         [2, 4], [2, 5], [3, 6], [3, 7],
                         [4, 8], [4, 9], [5, 10], [5, 11],
                         [8, 12], [8, 13], [10, 14], [10, 15],
                         [11, 16], [11, 17], [12, 18], [12, 19],
                         [13, 20], [13, 21], [14, 22], [14, 23],
                         [15, 24], [15, 25]
                             ]
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
        if triplets is not None:
            cmi2 = np.linspace(0, 1, len(triplet_combo))
            for j in range(len(triplets)):
                trips = triplets[j]
                for cm_ind, jp in zip(cmi2, triplet_combo):
                    ax.plot([pts[jp[0]][0], trips[jp[1]][0]], [pts[jp[0]][1], trips[jp[1]][1]],
                            linewidth=3.0, alpha=0.7, color=plt.cm.cool(cm_ind))
                    ax.scatter(trips[jp[1]][0], trips[jp[1]][1], s=20)
    return ax


def get_coords(coords, n=0, j=16):
    if not isinstance(coords, np.ndarray):
        coords = coords.asnumpy()
    best_joints = []
    for i in range(j):
        if np.amax(coords[n * j: (n + 1) * j][i]) == 0:
            best_joints.extend([0, 0])
        else:
            best = np.where(coords[n * j: (n + 1) * j][i] == np.amax(coords[n * j: (n + 1) * j][i]))
            best_joints.extend([best[1][0], best[0][0]])
    return best_joints


def get_and_plot(img, heatmap, triplet=None, n=0):
    if triplet is not None:
        joints = np.array(get_coords(heatmap))
        triplets = np.array(get_coords(triplet, n, j=26))
        display_coords(img, joints, triplets * 4)
    else:
        joints = np.array(get_coords(heatmap, n))
        display_coords(img, joints * 4)


def plot_plain(img, joints, triplets=None):
    joints = joints[0]
    if triplets is not None:
        display_coords(img, joints, triplets)
    else:
        display_coords(img, joints)
