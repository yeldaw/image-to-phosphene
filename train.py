from mpii_load import *
import plot
import os
import numpy as np
from mxnet.image import imread
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import gluon, autograd, gpu
from mxnet.gluon import nn, loss
from mxnet.gluon.data.vision import transforms
import time
import CustomDataset as CD
import mxnet.gluon.data.vision.datasets as datasets


def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return mx.nd.sum(output == label), mx.nd.sum(mx.nd.round(output) == label)


def plot_data(data, output, label):
    for i in range(len(data)):
        plot.display_coords(data[i], output[i])
        plot.display_coords(data[i], label[i])


class Network:

    def __init__(self, traindata, dir_, loss_, batch_size=64, testdata=[0]):
        self.traindata = traindata
        self.dir = dir_
        self.loss_func = loss_
        self.testdata = testdata
        self.batch_size = batch_size
        self.net = None
        self.trainer = None
        self.create_network()

    def create_network(self):
        self.net = nn.HybridSequential()
        self.net.hybridize(static_alloc=True, static_shape=True)
        self.net.add(
            # nn.BatchNorm(),
            # nn.Conv2D(16, (7, 7), activation='softrelu'),
            # nn.Conv1D(32, (5, 5), activation='softrelu'),
            # nn.MaxPool2D((2, 2)),
            # nn.Conv2D(64, (3, 3), activation='softrelu'),
            # nn.AvgPool2D((2, 2), strides=1),
            # nn.Conv2D(64, (5, 5), activation='softrelu'),
            # nn.MaxPool2D((2, 2), strides=3),
            # nn.Conv2D(64, (3, 3), activation='softrelu'),
            # nn.AvgPool2D((2, 2)),
            # nn.Conv2D(128, (5, 5), activation='softrelu'),
            # nn.Conv2D(32, (5, 5), activation='softrelu'),
            # nn.MaxPool2D((2, 2)),
            # nn.Dense(32, flatten=True)
            nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
            # nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Dense(200, activation="relu"),
            nn.Dense(80, activation="relu"),
            nn.Dense(32)
        )
        self.net.initialize(init=mx.init.Xavier(), ctx=gpu(0))
        self.trainer = gluon.Trainer(self.net.collect_params(), 'adam', {'learning_rate': 1E-2})

    def train_network(self, epochs=501):
        for epoch in range(1, epochs):
            train_loss, train_acc, adjusted_train_acc, test_acc, adjusted_test_acc = 0., 0., 0., 0., 0.
            tic = time.time()
            for train_data, train_label in self.traindata:
                # forward + backward
                train_data = train_data.as_in_context(gpu(0))
                train_label = train_label.as_in_context(gpu(0))
                with autograd.record():
                    train_output = self.net(train_data)
                    _loss = self.loss_func(train_output, train_label.astype('float32'))
                _loss.backward()
                # update parameters
                self.trainer.step(self.batch_size)
                # calculate training metrics
                train_loss += _loss.mean().asscalar()
                vals = acc(train_output, train_label)
                train_acc += vals[0].asscalar()
                adjusted_train_acc += vals[1].asscalar()
                mx.gpu(0).empty_cache()
                mx.nd.waitall()
            # calculate validation accuracy
            # for test_data, test_label in self.testdata:
            #     test_output = self.net(test_data)
            #     vals = acc(test_output, test_label)
            #     test_acc += vals[0].asscalar()
            #     adjusted_test_acc += vals[1].asscalar()
            #     mx.gpu(0).empty_cache()
            #     mx.nd.waitall()
            print(
                "Epoch %d: loss %.3f, train acc %.3f, adj train acc %.3f, test acc %.3f, "
                "adj test acc %.3f, in %.1f sec" % (epoch, train_loss / len(self.traindata),
                train_acc / len(self.traindata), adjusted_train_acc / len(self.traindata),
                test_acc / len(self.testdata), adjusted_test_acc / len(self.testdata), time.time() - tic))
            if epoch%10 == 0:
                self.net.export(self.dir + "Nets\\Netnet", epoch=epoch)


