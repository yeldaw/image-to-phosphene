import csv
from mpii_load import *
import os
import train
import matplotlib.pyplot as  plt


def update_image(image, data, loc, reverse=False):
    new_data = data[[i for i in data.keys()][0]]['annorect']
    joint_dict = new_data['annopoints']
    x_coords = [joint_dict[key]['x'] for key in joint_dict.keys()]
    y_coords = [joint_dict[key]['y'] for key in joint_dict.keys()]
    y1, y2, x1, x2 = [
        int(min(y_coords)),
        int(max(y_coords)),
        int(min(x_coords)),
        int(max(x_coords))
    ]
    coords = [
        y1 - 50 if y1 >= 50 else 0,
        y2 + 50 if y2 + 50 <= len(image) else len(image),
        x1 - 50 if x1 >= 50 else 0,
        x2 + 50 if x2 + 50 <= len(image[0]) else len(image[0])
    ]
    # new_image = image[coords[0]:coords[1], coords[2]:coords[3]]
    # plt.imsave(loc, new_image)
    new_data['annopoints'] = update_joints(joint_dict, coords[0], coords[2])
    if reverse:
        new_data['annopoints'] = rev_update_joints(joint_dict, coords[1], coords[3])
    data[[i for i in data.keys()][0]]['annorect'] = update_head(new_data, coords[0], coords[2])
    return data


def update_joints(data, y1, x1):
    for key in data.keys():
        data[key]['x'] = data[key]['x'] - x1
        data[key]['y'] = data[key]['y'] - y1
    return data


def rev_update_joints(data, y2, x2):
    for key in data.keys():
        data[key]['x'] = y2 - data[key]['x']
        data[key]['y'] = x2 - data[key]['y']
    return data


def update_head(data, y1, x1):
    data['x1'] = data['x1'] - x1
    data['x2'] = data['x2'] - x1
    data['y1'] = data['y1'] - y1
    data['y2'] = data['y2'] - y1
    return data


def pad_image(image, dir, x_size, y_size, save_dir):
    img_array = plt.imread(os.path.join(dir, image))
    if len(img_array) > x_size or len(img_array[0]) > y_size:
        return False
    zero_image = np.full((x_size, y_size, 3), 255)
    zero_image[:img_array.shape[0], :img_array.shape[1]] = np.flip(np.flip(img_array, 1), 0)
    new_name = image.strip('.jpg')[::-1] + '.jpg'
    if new_name == image:
        new_name = new_name.strip('.jpg') + '-1.jpg'
    plt.imsave(os.path.join(save_dir, new_name), zero_image.astype('uint8'))
    return True


image_dir = "F:\\Thesis Datasets\\mpii\\mpii_human_pose_v1\\"



# mat_file = "F:\\Thesis Datasets\\mpii_human_pose_v1_u12_2\\mpii_human_pose_v1_u12_1.mat"
# loaded_mat = loadmat(mat_file)
# cleanmat = clean_mat(loaded_mat)
# list_of_images = os.listdir(image_dir)
# image_set = train.create_dataset(image_dir, [[0, key] for key in cleanmat.keys()])
# create_json(cleanmat, image_dir, 'mpii_singular.json')


list_of_data = load_json(image_dir, 'mpii_singular.json')
for key in list_of_data.keys():
    list_of_data[key] = update_image(plt.imread(os.path.join(image_dir + "images\\" + key)), list_of_data[key],
              os.path.join(image_dir + "cut_images\\", key), reverse=True)

# create_json(list_of_data, image_dir, 'mpii_singular_updated.json')


# list_of_data = load_json(image_dir, 'mpii_singular_updated.json')
# padding = 512
# for key in list_of_data.keys():
#     padded = pad_image(key, image_dir + "cut_images\\", padding, padding, image_dir + f"extended_images_{padding}_white\\")
#     # if padded:
#     #     train.display(image_dir + "padded_images\\", key, list_of_data[key])
#     print(key)
