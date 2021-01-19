from mxnet.gluon import nn
from mxnet import nd
from mxnet import gpu
import matplotlib.pyplot as plt
import plot
from mxnet.gluon.data.vision import transforms
import warnings
import os
import numpy as np

root_dir = "F:\\Thesis Datasets\\mpii\\mpii_human_pose_v1\\"


image_size = 256
image_dir = f"full_{image_size}\\"

# model_num = '4'
# model_file = f"Nets\\simple_model{model_num}.params"


def test(net):
    files = ['test.jpg', 'test2.jpg', 'test4.jpg']
    for f in files:
        transformer = transforms.Compose([transforms.ToTensor()])
        # for file in os.listdir(root_dir + image_dir):
        # file = root_dir + 'test_256\\' + f
        c = plt.imread(root_dir + 'test_256\\' + f)
        test_arr = nd.array(c).as_in_context(gpu(0)).reshape(1, image_size, image_size, 3)
        transformed_arr = transformer(test_arr)
        hm_preds = net(transformed_arr)[0]
        coords = plot.get_and_plot(transformed_arr[0], hm_preds, 7)
        # plot.display_coords(transformed_arr[0], np.array(coords) * 4)
        plt.show()


# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     deserialized_net = nn.SymbolBlock.imports(root_dir + "Network_export\\Epoch_8_stack_no_weights-final-symbol.json", ['data'],
#                                               root_dir + "Network_export\\Epoch_8_stack_no_weights-final-4260.params", ctx=gpu(0))
#
#     test(deserialized_net)


