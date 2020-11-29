from mpii_load import *
import train
import os
import numpy as np
from mxnet.image import imread
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
import time
import CustomDataset as CD
#
#
# # img = imread('F:\\Thesis Datasets\\test\\Untitled.png', flag=0)
# label = {
#     '6':
#         {'x': 10, 'y': 15, 'name': 'pelvis', 'is_visible': 0}
# }
#
# file = 'F:\\Thesis Datasets\\test\\'
# npz = 'file.npz'
#
# npzfile = np.load(file + npz)
# traindata = [[value, key + ".png"] for key, value in dict(npzfile).items()]

transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.13, 0.31)])

batch_size = 256

def load_dataset(batch_size=None):
    dataset = CD.CustomDataset('F:\\Thesis Datasets\\test', 'data.npz', 'label.npz', 5, 5)
    transformed_dataset = dataset.transform_first(transformer)
    traindata = mx.gluon.data.DataLoader(transformed_dataset, batch_size=10)
    return traindata


# train_data = aaa(traindata)
train_data = load_dataset(batch_size)
net = nn.Sequential()
net.add(
    nn.Dense(25, flatten=True)
)
net.initialize(init=mx.init.Xavier())

sigmoid_cross_entropy = CD.SigmoidBinaryCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})


def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    print('a')
    return (output.argmax(axis=1) ==
            label.astype('float32')).mean().asscalar()


for epoch in range(1000):
    train_loss, train_acc, valid_acc = 0., 0., 0.
    tic = time.time()
    for data, label in train_data:
        # forward + backward
        with autograd.record():
            output = net(data)
            loss = sigmoid_cross_entropy(output, label.astype('float32'))
        loss.backward()
        # update parameters
        trainer.step(batch_size)
        # calculate training metrics
        train_loss += loss.mean().asscalar()
        # train_acc += acc(output, label)
    # calculate validation accuracy
    print("Epoch %d: loss %.3f, train acc %.3f, in %.1f sec" % (
            epoch, train_loss/len(train_data), train_acc/len(train_data),
            time.time()-tic))
# layer = nn.Dense(len(label.keys()))
# for d, l in traindata:
#     output = layer(d)
#
# print("Hello")
# image_dir = "F:\\Thesis Datasets\\mpii\\mpii_human_pose_v1\\images\\"
# mat_file = "F:\\Thesis Datasets\\mpii_human_pose_v1_u12_2\\mpii_human_pose_v1_u12_1.mat"


# loaded_mat = loadmat(mat_file)
# cleanmat = clean_mat(loaded_mat)
# list_of_images = os.listdir(image_dir)
# transform_train = SimplePoseDefaultTrainTransform(num_joints=)
# image_set = train.create_dataset(image_dir, [[0, key] for key in cleanmat.keys()])
# train.create_nn(image_set, cleanmat)

# for image_name in cleanmat.keys():
#     train.display(image_dir, image_name, cleanmat[image_name])


# data = {
#     '4,1': {
#         'image_name':       '015601864.jpg',
#         'annorect':         {
#             'x1':   841,
#             'y1':   145,
#             'x2':   902,
#             'y2':   228,
#             'scale':    2.472116502109073,
#             'annopoints': {
#                 '6':
#                     {'x': 979, 'y': 221, 'name': 'pelvis', 'is_visible': 0},
#                 '7':
#                     {'x': 906, 'y': 190, 'name': 'thorax', 'is_visible': 0},
#                 '8':
#                     {'x': 912.4915, 'y': 190.6586, 'name': 'upper neck', 'is_visible': None},
#                 '9':
#                     {'x': 830.5085, 'y': 182.3414, 'name': 'head top', 'is_visible': None},
#                 '0':
#                     {'x': 895, 'y': 293, 'name': 'right ankle', 'is_visible': 1},
#                 '1':
#                     {'x': 910, 'y': 279, 'name': 'right knee', 'is_visible': 1},
#                 '2':
#                     {'x': 945, 'y': 223, 'name': 'right hip', 'is_visible': 0},
#                 '3':
#                     {'x': 1012, 'y': 218, 'name': 'left hip', 'is_visible': 1},
#                 '4':
#                     {'x': 961, 'y': 315, 'name': 'left knee', 'is_visible': 1},
#                 '5':
#                     {'x': 960, 'y': 403, 'name': 'left ankle', 'is_visible': 1},
#                 '10':
#                     {'x': 871, 'y': 304, 'name': 'right wrist', 'is_visible': 1},
#                 '11':
#                     {'x': 883, 'y': 229, 'name': 'right elbow', 'is_visible': 1},
#                 '12':
#                     {'x': 888, 'y': 174, 'name': 'right shoulder', 'is_visible': 0},
#                 '13':
#                     {'x': 924, 'y': 206, 'name': 'left shoulder', 'is_visible': 1},
#                 '14':
#                     {'x': 1013, 'y': 203, 'name': 'left elbow', 'is_visible': 1},
#                 '15':
#                     {'x': 955, 'y': 263, 'name': 'left wrist', 'is_visible': 1}}
#
#         },
#         'img_train':        1,
#         'single_person':    np.array((1, 2)),
#         'act':              {
#             'act_id': 1, 'act_name': 'curling', 'cat_name': 'sports'
#         },
#         'video_list':   'aAOusnrSsHI'
#     }
# }
# train.display('F:\\Thesis Datasets\\mpii\\mpii_human_pose_v1\\images\\', '015601864.jpg', data)