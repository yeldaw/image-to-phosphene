from mxnet import gluon


class UpSample(gluon.HybridBlock):
    def __init__(self, scale, sample_type):
        super(UpSample, self).__init__()
        self.scale = scale
        self.sample_type = sample_type

    def hybrid_forward(self, F, x):
        return F.UpSampling(x, scale=self.scale, sample_type=self.sample_type)
