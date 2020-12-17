from mxnet.gluon import nn
from mxnet import nd
from mxnet import gpu
import matplotlib.pyplot as plt
import plot
from mxnet.gluon.data.vision import transforms
import warnings
import os

root_dir = "F:\\Thesis Datasets\\mpii\\mpii_human_pose_v1\\"


image_size = 256
image_dir = f"test_{image_size}\\"

# model_num = '4'
# model_file = f"Nets\\simple_model{model_num}.params"


def test(net):
    transformer = transforms.Compose([transforms.ToTensor()])
    for file in os.listdir(root_dir + image_dir):
        c = plt.imread(root_dir + image_dir + file)
        test_arr = nd.array(c).as_in_context(gpu(0)).reshape(1, image_size, image_size, 3)
        transformed_arr = transformer(test_arr)
        coords = net(transformed_arr)
        plot.display_coords(transformed_arr[0], coords[0])
        plt.show()


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    deserialized_net = nn.SymbolBlock.imports(root_dir + "Network_export\\Epoch-symbol.json", ['data'],
                                              root_dir + "Network_export\\Epoch-0053.params", ctx=gpu(0))

    test(deserialized_net)
