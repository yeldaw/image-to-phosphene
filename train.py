import plot
import mxnet as mx
from mxnet import gluon, autograd, gpu
import time
import MySequential as ms
import mxnet.ndarray as nd
from mxnet.gluon import nn
import warnings


def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return mx.nd.sum(output == label), mx.nd.sum(mx.nd.round(output) == label)


def plot_data(data, output, label):
    for i in range(len(data)):
        plot.display_coords(data[i], output[i])
        plot.display_coords(data[i], label[i])


class Network:

    def __init__(self, train_dataset, dir_, batch_size=4, testdata=[0], train_old_net=False, epoch=1, triplet=False):
        self.train_dataset = train_dataset
        self.traindata = mx.gluon.data.DataLoader(self.train_dataset, batch_size=batch_size)
        self.dir = dir_
        self.testdata = testdata
        self.batch_size = batch_size
        self.net = None
        self.trainer = None
        self.triplet = triplet
        if train_old_net:
            self.load_network(epoch)
            self.epoch = epoch
            self.create_trainer()
            self.trainer.load_states(f"{self.dir}Final_export\\Final-joint-8stack-fix-optimizer-{epoch:>04}")
        else:
            self.create_network()
            self.epoch = 0
            self.create_trainer()

    def create_network(self):
        if self.triplet:
            self.net = ms.MySequential(output_dim=26, nstack=2, triplet=self.triplet)
        else:
            self.net = ms.MySequential(nstack=8, triplet=self.triplet)
        self.net.hybridize(static_alloc=True, static_shape=True)
        self.net.initialize(init=mx.init.Xavier(), ctx=gpu(0))

    def load_network(self, epoch):
        if self.triplet:
            self.net = ms.MySequential(output_dim=26, nstack=2, triplet=self.triplet)
            self.net.load_parameters(self.dir + f"Final_export\\Final_triplet_8stack-{epoch:>04}.params", ctx=gpu(0))
        else:
            self.net = ms.MySequential(nstack=8, triplet=self.triplet)
            self.net.load_parameters(self.dir + f"Final_export\\Final-joint-8stack-fix-{epoch:>04}.params", ctx=gpu(0))

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     self.net = nn.SymbolBlock.imports(self.dir + "Network_export\\Epoch_8_stack_no_weights-symbol.json", ['data'],
        #                                       self.dir + f"Network_export\\Epoch_8_stack_no_weights-{epoch}.params", ctx=gpu(0))
        self.net.hybridize(static_alloc=True, static_shape=True)

    def create_trainer(self):
        schedule = mx.lr_scheduler.MultiFactorScheduler(step=[500, 1000, 2000, 5000], factor=0.5)
        schedule.base_lr = 1
        optimizer = mx.optimizer.Adam(lr_scheduler=schedule)
        self.trainer = gluon.Trainer(self.net.collect_params(), optimizer)

    def train_network(self, epochs=5001):
        for epoch in range(int(self.epoch) + 1, epochs):
            train_loss, train_acc, adjusted_train_acc, test_acc, adjusted_test_acc = 0., 0., 0., 0., 0.
            loss_per_hg = []
            tic = time.time()
            for train_data, train_label, train_weight in self.traindata:
                # train_label = train_label.as_in_context(gpu(0))
                with autograd.record():
                    hm_preds = self.net(train_data)
                    # a = plot.get_coords(hm_preds[0])
                    _loss = self.net.calc_loss(hm_preds, train_label, train_weight)
                _loss.backward()
                # update parameters
                self.trainer.step(self.batch_size)
                # calculate training metrics
                train_loss += _loss.mean().asscalar()
                for i in range(int(len(_loss)/self.batch_size)):
                    loss_per_hg.append(_loss[i:(i + 1)*self.batch_size].mean().asscalar())
                # vals = acc(train_output, train_label)
                # train_acc += vals[0].asscalar()
                # adjusted_train_acc += vals[1].asscalar()
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
            print(loss_per_hg)
            if epoch % 50 == 0:
                self.net.export(self.dir + "Final_export\\Final-joint-8stack-fix", epoch=epoch)
            if epoch % 250 == 0:
                self.trainer.save_states(f"{self.dir}Final_export\\Final-joint-8stack-fix-optimizer-{epoch:>04}")
            self.train_dataset.get_data()
            # self.traindata = mx.gluon.data.DataLoader(self.train_dataset, batch_size=self.batch_size)

    def train_triplet(self, epochs=5001):
        for epoch in range(int(self.epoch) + 1, epochs):
            train_loss, train_acc, adjusted_train_acc, test_acc, adjusted_test_acc = 0., 0., 0., 0., 0.
            tic = time.time()
            for train_data, train_label, train_weight, triplet_data in self.traindata:
                with autograd.record():
                    triplet_maps = nd.Concat(train_data, train_label)
                    hm_preds = self.net(triplet_maps)
                    _loss = self.net.calc_triplet_loss(hm_preds, triplet_data)
                _loss.backward()
                self.trainer.step(self.batch_size)
                train_loss += _loss.mean().asscalar()
                mx.gpu(0).empty_cache()
                mx.nd.waitall()
            print(
                f"Epoch {epoch}: loss {train_loss / len(self.traindata):.3f}, "
                f"train acc {train_acc / len(self.traindata):.3f},  in {time.time() - tic:.3f} sec")
            if epoch % 50 == 0:
                self.net.export(self.dir + "Final_export\\Final_triplet_2stack", epoch=epoch)
            if epoch % 250 == 0:
                self.trainer.save_states(f"{self.dir}Final_export\\2stack_triplet_optimizer-{epoch:>04-}")
            self.train_dataset.get_data()
            # if epoch % 5 == 0:
            #     self.train_dataset.get_triplet()
            #     self.traindata = mx.gluon.data.DataLoader(self.train_dataset, batch_size=self.batch_size)

