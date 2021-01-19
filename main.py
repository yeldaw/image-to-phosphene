from mpii_load import *
import train
import plot
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet.gluon.data.vision import transforms
import CustomDataset as CD


# Maximum image size
image_size = 256
batch_size = 4
root_dir = "F:\\Thesis Datasets\\mpii\\mpii_human_pose_v1\\"

image_dir = f"full_256\\"
# image_dir = f"cut_images_{image_size}_white\\"
# image_dir = f"images\\"
#
# test_dir = f"test_{image_size}\\"
label_json = "mpii_full_256"
# label_json = "mpii_256_shrunk_extended"
# label_json = "mpii_singular"
modified_images_dir = ''


def modify_json():
    original = load_json(root_dir, label_json + ".json")
    new = reverse_dic(original)
    with open(os.path.join(root_dir, label_json + "_rev2.json"), 'w+') as infile:
        json.dump(new, infile)


def modify_images():
    images = os.listdir(root_dir + image_dir)
    for image in images:
        img = plt.imread(root_dir + image_dir + image)
        new_name = image.strip('.jpg')[::-1] + '.jpg'
        if new_name == image:
            new_name = new_name.strip('.jpg') + '-1.jpg'
        plt.imsave(root_dir + modified_images_dir + new_name, np.flip(np.flip(img, 1), 0))


def train_setup(net=False):
    # Creates transformer
    transformer = transforms.Compose([transforms.ToTensor()])

    # Loads datasets
    train_dataset = CD.CustomDataset(root_dir, image_dir, label_json + ".json",
                                     image_size, image_size, 64, 64, transform=transformer)
    # test_dataset = CD.CustomDataset(root_dir, test_dir, label_json + ".json",
    # image_size, image_size, transform=transformer)

    # testdata = mx.gluon.data.DataLoader(test_dataset, batch_size=batch_size)

    net = train.Network(train_dataset, root_dir, batch_size, train_old_net=net, epoch="2600")
    net.train_network()


def test(dic, k1):
    k2 = k1.strip('.jpg')[::-1] + '.jpg'
    a = CD.grab_joints(dic[k1])
    b = CD.grab_joints(dic[k2])
    plot.display_coords(plt.imread(root_dir + image_dir + k1), a)
    plot.display_coords(plt.imread(root_dir + image_dir + k2), b)


train_setup(True)
# modify_json()


# first = load_json(root_dir, label_json + "_extended.json")
# reverse_dic(first, root_dir + 'cut_images_256\\')
# first = load_json(root_dir, label_json + ".json")
# second = load_json(root_dir, label_json + "_rev2.json")
# first.update(second)
# for key in first.keys():
#     test(first, key)
# with open(os.path.join(root_dir, label_json + "_extended.json"), 'w+') as infile:
#     json.dump(first, infile)
