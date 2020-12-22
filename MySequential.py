from mxnet.gluon import nn
import mxnet.symbol as sym
from mxnet.ndarray import Concat
from mxnet import nd


class MySequential(nn.HybridBlock):
    def __init__(self, initial=0, output=1, prefix=None, params=None):
        super(MySequential, self).__init__(prefix=prefix, params=params)
        self.initial = initial
        self.output = output
        self.net_length = 0
        self.single_length = 0

    def add(self, *blocks):
        """Adds block on top of the stack."""
        for block in range(len(blocks)):
            self.register_child(blocks[block])
        self.net_length = len(self._children) - self.initial - self.output
        if self.net_length%2 == 0:
            self.single_length = int(self.net_length/2)
        else:
            raise OutputLayerMissingError("""Your downsampling and upsampling layers are mismatched""")

    def hybrid_forward(self, F, x):
        for layer in range(self.initial):
            x = self._children[str(layer)](x)
        results = {str(self.initial - 1): x}
        for layer in range(self.initial,  self.single_length + self.initial):
            results[str(layer)] = self._children[str(layer)](results[str(layer - 1)])
        x = results[str(self.single_length + self.initial - 1)]
        for layer in range(self.initial + self.single_length, 2 * self.single_length + self.initial):
            # x = self._children[str(layer)](x)
            x = sym.Concat(self._children[str(layer)](x), results[str(len(self._children) - self.output - layer + self.initial - 2)])
        for layer in range(self.output):
            x = self._children[str(self.net_length + self.initial + layer)](x)
        # results2 = {'-1': results[str(self.net_length - 1)].as_in_context(gpu(0))}
        # for layer in range(self.net_length):
        #     results2[str(layer)] = self._children[layer + self.net_length](results2[str(layer - 1)]) + \
        #                            results[str(self.net_length - layer - 2)]
        # for block in self.blocks.keys():
        #     self.results[self.blocks[block]] = self._children[block](self.outputs[block - 1])
        # x = self.outputs[self.blocks[block]]
        # for block in self.outputs.keys():
        #     x = self._children[block](self.outputs[block]) + self.
        return x

    def __len__(self):
        return len(self._children)


class Residual(nn.HybridBlock):
    """The Residual block of ResNet."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nd.relu(Y + X)


class BatchNormCudnnOff(nn.BatchNorm):
    """Batch normalization layer without CUDNN. It is a temporary solution.

    Parameters
    ----------
    kwargs : arguments goes to mxnet.gluon.nn.BatchNorm
    """
    def __init__(self, **kwargs):
        super(BatchNormCudnnOff, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, gamma, beta, running_mean, running_var):
        return F.BatchNorm(x, gamma, beta, running_mean, running_var,
                           name='fwd', cudnn_off=True, **self._kwargs)


class OutputLayerMissingError(Exception):
    pass
