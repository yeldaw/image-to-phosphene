from mxnet.gluon import nn, loss
import mxnet.symbol as sym
from mxnet.ndarray import Concat
from mxnet import nd

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


class UpSample(nn.HybridBlock):
    def __init__(self, scale, sample_type):
        super(UpSample, self).__init__()
        self.scale = scale
        self.sample_type = sample_type

    def hybrid_forward(self, F, x):
        return F.UpSampling(x, scale=self.scale, sample_type=self.sample_type)


class Residual(nn.HybridBlock):
    """The Residual block of ResNet."""

    def __init__(self, input_size, output_size, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.relu = nn.Activation('relu')
        self.bn1 = nn.BatchNorm()
        self.conv1 = nn.Conv2D(int(output_size/2), (1, 1))
        self.bn2 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(int(output_size/2), (3, 3), padding=1)
        self.bn3 = nn.BatchNorm()
        self.conv3 = nn.Conv2D(output_size, (1, 1))
        if input_size == output_size:
            self.skip = False
        else:
            self.skip_layer = nn.Conv2D(output_size, (1, 1))
            self.skip = True

    def hybrid_forward(self, F, x, *args, **kwargs):
        if self.skip:
            res = self.skip_layer(x)
        else:
            res = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = out + res
        return out


class Hourglass(nn.HybridBlock):

    def __init__(self, num, dim, increase=0, prefix=None, params=None):
        super(Hourglass, self).__init__(prefix=prefix, params=params)
        new_dim = dim + increase
        self.up1 = Residual(dim, dim)
        # Lower branch
        self.pool1 = nn.MaxPool2D(2, 2)
        self.low1 = Residual(dim, new_dim)
        # Recursive hourglass
        if num > 1:
            self.low2 = Hourglass(num-1, new_dim)
        else:
            self.low2 = Residual(new_dim, new_dim)
        self.low3 = Residual(new_dim, dim)
        self.up2 = UpSample(scale=2, sample_type='nearest')

    def hybrid_forward(self, F, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class Initial(nn.HybridBlock):

    def __init__(self, input_dim, prefix=None, params=None):
        super(Initial, self).__init__(prefix=prefix, params=params)
        self.conv1 = Conv(64, kernel_size=7, strides=2, bn=True, relu=True)
        self.res1 = Residual(64, 128)
        self.maxp1 = nn.MaxPool2D(2, 2)
        self.res2 = Residual(128, 128)
        self.res3 = Residual(128, input_dim)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.maxp1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x


class Conv(nn.HybridBlock):

    def __init__(self, output_size, kernel_size=3, strides=1, bn=False, relu=True, prefix=None, params=None):
        super(Conv, self).__init__(prefix=prefix, params=params)
        self.conv = nn.Conv2D(output_size, kernel_size, strides=strides, padding=(kernel_size-1)//2)
        self.bn = None
        self.relu = None
        if bn:
            self.bn = nn.BatchNorm()
        if relu:
            self.act = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.act(x)
        return x


class Features(nn.HybridBlock):

    def __init__(self, input_size, bn=False, relu=False, prefix=None, params=None):
        super(Features, self).__init__(prefix=prefix, params=params)
        self.res = Residual(input_size, input_size)
        self.conv = Conv(input_size, kernel_size=1, bn=bn, relu=relu)

    def hybrid_forward(self, F, x):
        x = self.res(x)
        x = self.conv(x)
        return x


class HeatmapLoss(nn.HybridBlock):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def hybrid_forward(self, F, input, output):
        l = ((input - output)**2)
        l = l.mean(axis=3).mean(axis=2).mean(axis=1)
        return l


class MySequential(nn.HybridBlock):
    def __init__(self, input_dim=256, output_dim=16, nstack=1, prefix=None, params=None):
        super(MySequential, self).__init__(prefix=prefix, params=params)
        self.nstack = nstack
        self.initial = Initial(input_dim)

        self.hg = nn.HybridSequential()
        self.hg.add(*[Hourglass(4, input_dim) for n in range(nstack)])

        self.feature = nn.HybridSequential()
        self.feature.add(*[Features(input_dim, bn=True, relu=True) for n in range(nstack)])

        self.preds = nn.HybridSequential()
        self.preds.add(*[nn.Conv2D(output_dim, (1, 1)) for n in range(nstack)])

        self.merge_preds = nn.HybridSequential()
        self.merge_preds.add(*[nn.Conv2D(input_dim, (1, 1)) for n in range(nstack-1)])
        self.merge_features = nn.HybridSequential()
        self.merge_features.add(*[nn.Conv2D(input_dim, (1, 1)) for n in range(nstack-1)])

        self.loss_func = HeatmapLoss()

    def add(self, *blocks):
        """Adds block on top of the stack."""
        for block in blocks:
            self.register_child(block)

    def hybrid_forward(self, F, x):
        x = self.initial(x)
        comb_preds = []
        for i in range(0, self.nstack):
            hg = self.hg[i](x)
            feature = self.feature[i](hg)
            preds = self.preds[i](feature)
            comb_preds.append(preds)
            if i != self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return Concat(*comb_preds, dim=1)

    def calc_loss(self, comb_preds, heatmaps):
        if not isinstance(comb_preds, list):
            comb_preds = [comb_preds]
        combined_loss = []
        for i in range(self.nstack):
            combined_loss.append(self.loss_func(comb_preds[0][:, i * 16: (i + 1) * 16], heatmaps))
        combined_loss = Concat(*combined_loss, dim=0)
        return combined_loss

    def __len__(self):
        return len(self._children)
