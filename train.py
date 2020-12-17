import plot
import mxnet as mx
from mxnet import gluon, autograd, gpu, nd
from mxnet.gluon import nn
import time
import CustomBlocks as CB
from MySequential import MySequential

def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return mx.nd.sum(output == label), mx.nd.sum(mx.nd.round(output) == label)


def plot_data(data, output, label):
    for i in range(len(data)):
        plot.display_coords(data[i], output[i])
        plot.display_coords(data[i], label[i])


class Network:

    def __init__(self, traindata, dir_, loss_, batch_size=4, testdata=[0]):
        self.traindata = traindata
        self.dir = dir_
        self.loss_func = loss_
        self.testdata = testdata
        self.batch_size = batch_size
        self.net = None
        self.net_length = None
        self.trainer = None
        self.create_network()

    def create_network(self):
        self.net = MySequential(9)
        self.net.hybridize(static_alloc=True, static_shape=True)
        self.net.add(
            # Initial
            nn.Conv2D(8, (7, 7)),
            nn.Conv2D(16, (5, 5)),
            nn.Conv2D(32, (3, 3)),
            nn.MaxPool2D((2, 2)),
            nn.Conv2D(64, (3, 3)),
            nn.Conv2D(32, (7, 7)),
            nn.Conv2D(64, (5, 5)),
            nn.Conv2D(32, (7, 7)),
            nn.Conv2D(64, (3, 3)),

            # First part of the hourglass
            nn.Conv2D(16, (3, 3), activation='softrelu'),
            nn.Conv2D(32, (5, 5), activation='softrelu'),
            nn.MaxPool2D((2, 2)),
            nn.Conv2D(64, (3, 3), activation='softrelu'),
            nn.Conv2D(128, (5, 5), activation='softrelu'),
            nn.MaxPool2D((3, 3)),
            # nn.MaxPool2D((2, 2)),
            # nn.Conv2D(64, (4, 4), activation='softrelu'),
            # nn.MaxPool2D((2, 2)),
            # nn.Conv2D(64, (3, 3), activation='softrelu'),
            # nn.Conv2D(64, (5, 5), activation='softrelu'),
            # nn.MaxPool2D((2, 2), strides=3),

            #Second part of the hourglass
            # CB.UpSample(scale=3, sample_type='nearest'),
            # nn.Conv2DTranspose(64, (5, 5)),
            # nn.Conv2DTranspose(64, (3, 3)),
            # CB.UpSample(scale=2, sample_type='nearest'),
            # nn.Conv2DTranspose(64, (4, 4)),
            # CB.UpSample(scale=2, sample_type='nearest'),
            CB.UpSample(scale=3, sample_type='nearest' ),
            nn.Conv2DTranspose(32, (5, 5)),
            nn.Conv2DTranspose(16, (3, 3), activation='softrelu'),
            CB.UpSample(scale=2, sample_type='nearest'),
            nn.Conv2DTranspose(16, (5, 5), activation='softrelu'),
            nn.Conv2DTranspose(3, (3, 3), activation='softrelu'),
            nn.Dense(16384)
        )
        self.net.initialize(init=mx.init.Xavier(), ctx=gpu(0))
        self.trainer = gluon.Trainer(self.net.collect_params(), 'adam', {'learning_rate': 1E-3})
        self.net_length = len(self.net)

    def train_network(self, epochs=501):
        for epoch in range(1, epochs):
            train_loss, train_acc, adjusted_train_acc, test_acc, adjusted_test_acc = 0., 0., 0., 0., 0.
            tic = time.time()
            for train_data, train_label in self.traindata:
                train_label = train_label.as_in_context(gpu(0))
                with autograd.record():
                    train_output = self.net(train_data)
                    train_output = train_output.reshape(16, 32, 32)
                    # train_output = self.net(train_data)
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
                f"Epoch {epoch}: loss {train_loss / len(self.traindata):.3f}, "
                f"train acc {train_acc / len(self.traindata):.3f},  in {time.time() - tic:.3f} sec")
            if epoch % 10 == 0:
                self.net.export(self.dir + "Network_export\\Epoch", epoch=epoch)
