from mxnet.gluon import nn
from mxnet import nd
from mxnet import gpu
import matplotlib.pyplot as plt
import plot
from mxnet.gluon.data.vision import transforms
import warnings
import os
import numpy as np
import cv2


root_dir = "F:\\Thesis Datasets\\mpii\\mpii_human_pose_v1\\"
image_size = 256
image_dir = f"full_{image_size}\\"

# model_num = '4'
# model_file = f"Nets\\simple_model{model_num}.params"


def test(net):
    for f in os.listdir(f"{root_dir}test_images\\"):
        transformer = transforms.Compose([transforms.ToTensor()])
        c = plt.imread(f"{root_dir}test_images\\{f}")
        factor_y = c.shape[0]/256
        if factor_y > 1:
            c = cv2.resize(c, (int(c.shape[1]/factor_y), 256), interpolation=cv2.INTER_AREA)
        factor_x = c.shape[1] / 256
        if factor_x > 1:
            c = cv2.resize(c, (256, int(c.shape[0])/factor_x), interpolation=cv2.INTER_AREA)
        zero_image = np.full((256, 256, 3), 255)
        zero_image[:c.shape[0], :c.shape[1]] = c

        test_arr = nd.array(zero_image).as_in_context(gpu(0)).reshape(1, image_size, image_size, 3)
        transformed_arr = transformer(test_arr)
        hm_preds = net(transformed_arr)[0]
        coords = plot.get_and_plot(transformed_arr[0], hm_preds, n=7)
        # plot.display_coords(transformed_arr[0], np.array(coords) * 4)
        plt.show()
        bla = input()
        if bla == 'y':
            return hm_preds, zero_image


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    deserialized_net = nn.SymbolBlock.imports(root_dir + "Network_export\\Epoch_8_stack_no_weights-symbol.json", ['data'],
                                              root_dir + "Network_export\\Epoch_8_stack_no_weights-3900.params", ctx=gpu(0))

    results = test(deserialized_net)

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     deserialized_net = nn.SymbolBlock.imports(root_dir + "Network_export\\Triplet-test-symbol.json", ['data'],
#                                               root_dir + "Network_export\\Triplet-test-0550.params", ctx=gpu(0))
#     test(deserialized_net)
#

#550
