from mxnet.gluon import nn


class MySequential(nn.HybridBlock):
    def __init__(self, prefix=None, params=None):
        super(MySequential, self).__init__(prefix=prefix, params=params)
        self.net_length = 0
        self.single_length = 0

    def add(self, *blocks):
        """Adds block on top of the stack."""
        for block in range(len(blocks)):
            self.register_child(blocks[block])
        self.net_length = len(self._children)
        if self.net_length%2 != 0:
            self.single_length = int((self.net_length - 1) / 2)
        else:
            raise OutputLayerMissingError("""You have either not defined an output layer, 
            or the downsampling and upsampling layers are mismatched""")

    def hybrid_forward(self, F, x):
        results = {'-1': x}
        for layer in range(self.single_length):
            results[str(layer)] = self._children[str(layer)](results[str(layer - 1)])
        x = results[str(self.single_length - 1)]
        for layer in range(self.single_length):
            x = self._children[str(layer + self.single_length)](x) + results[str(self.single_length - layer - 2)]
        x = self._children[str(self.net_length - 1)](x)
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


class OutputLayerMissingError(Exception):
    pass
