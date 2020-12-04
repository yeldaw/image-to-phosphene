from mpii_load import *
import train
import plot
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet.gluon import loss
from mxnet.gluon.data.vision import transforms
import CustomDataset as CD


# Maximum image size
image_size = 512
batch_size = 2
root_dir = "F:\\Thesis Datasets\\mpii\\mpii_human_pose_v1\\"

image_dir = f"padded_images_{image_size}_white\\"
test_dir = f"test_{image_size}\\"
label_json = "mpii_singular_updated.json"
modified_images_dir = ''


def modify_json():
    original = load_json(root_dir, label_json)
    # sample = {'060111501.jpg': original['060111501.jpg']}
    modify_images()
    reverse_dic(original, image_size, image_size)
    print(0)
    # with open(os.path.join(root_dir, mirror_json), 'w+') as infile:
    #     json.dump(original, infile)


def modify_images():
    images = os.listdir(root_dir + image_dir)
    for image in images:
        img = plt.imread(root_dir + image_dir + image)
        new_name = image.strip('.jpg')[::-1] + '.jpg'
        if new_name == image:
            new_name = new_name.strip('.jpg') + '-1.jpg'
        plt.imsave(root_dir + modified_images_dir + new_name, np.flip(np.flip(img, 1), 0))


def train_setup():
    # Creates transformer
    transformer = transforms.Compose([transforms.ToTensor()])
    # transformed_dataset = train_dataset.transform_first(transformer)

    # Loads datasets
    train_dataset = CD.CustomDataset(root_dir, image_dir, label_json, image_size, image_size, transform=transformer)
    test_dataset = CD.CustomDataset(root_dir, test_dir, label_json, image_size, image_size, transform=transformer)

    traindata = mx.gluon.data.DataLoader(train_dataset, batch_size=batch_size)
    testdata = mx.gluon.data.DataLoader(test_dataset, batch_size=batch_size)

    l2 = loss.L2Loss()
    net = train.Network(traindata, root_dir, l2, batch_size, testdata)
    net.train_network()


train_setup()
# modify_json()
