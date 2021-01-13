from mpii_load import *
import os
import matplotlib.pyplot as plt
import cv2
import plot

def update_image(image, data, shrink=False):
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
    pad = [True if y1 >= 50 else False, True if y2 + 50 <= len(image) else False,
           True if x1 >= 50 else False, True if x2 + 50 <= len(image[0]) else False]
    coords = [
        y1 - 50 if pad[0] else 0,
        y2 + 50 if pad[1] else len(image),
        x1 - 50 if pad[2] else 0,
        x2 + 50 if pad[3] else len(image[0])
    ]
    # new_image = image[coords[0]:coords[1], coords[2]:coords[3]]
    # plt.imsave(loc, new_image)
    new_data['annopoints'] = update_joints(joint_dict, coords[0], coords[2], shrink)
    data[[i for i in data.keys()][0]]['annorect'] = update_head(new_data, coords[0], coords[2], shrink)
    return data


def update_joints(data, y1, x1, shrink):
    for key in data.keys():
        data[key]['x'] = data[key]['x'] - x1
        data[key]['y'] = data[key]['y'] - y1
    if shrink:
        rescale_joints(data)
    return data


def update_head(data, y1, x1):
    """
    Head shrink, head reverse not implemented
    :param data:
    :param y1:
    :param x1:
    :return:
    """
    data['x1'] = data['x1'] - x1
    data['x2'] = data['x2'] - x1
    data['y1'] = data['y1'] - y1
    data['y2'] = data['y2'] - y1
    return data


def pad_image(image, image_dir, x_size, y_size, save_dir):
    img_array = plt.imread(os.path.join(image_dir, image))
    if len(img_array) > x_size or len(img_array[0]) > y_size:
        return False
    zero_image = np.full((x_size, y_size, 3), 255)
    zero_image[:img_array.shape[0], :img_array.shape[1]] = img_array
    # zero_image[:img_array.shape[0], :img_array.shape[1]] = np.flip(np.flip(img_array, 1), 0)
    # new_name = image.strip('.jpg')[::-1] + '.jpg'
    # if new_name == image:
    #     new_name = new_name.strip('.jpg') + '-1.jpg'
    plt.imsave(os.path.join(save_dir, image), zero_image.astype('uint8'))
    return True


def rescale_image(image, scale_value, image_dir, save_dir):
    import cv2

    img = cv2.imread(image_dir + image, cv2.IMREAD_UNCHANGED)
    if (img.shape[0] > 256 or img.shape[1] > 256) and (img.shape[0] < 512 and img.shape[1] < 512):
        width = int(img.shape[1] * scale_value / 100)
        height = int(img.shape[0] * scale_value / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(save_dir + image, resized)
        # print('Resized Dimensions : ', resized.shape)

        # cv2.imshow("Resized image", resized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def rescale_joints(data):
    for key in data:
        data[key]['x'] = int(data[key]['x']/2)
        data[key]['y'] = int(data[key]['y']/2)
    return data


def full_image(image, data, scale=256, extra=None):
    # new_img = cv2.resize(image, (scale, scale), interpolation=cv2.INTER_AREA)
    new_data = data[[i for i in data.keys()][0]]
    joint_dict = new_data['annorect']['annopoints']
    for key in joint_dict.keys():
        joint_dict[key]['x'] = int(joint_dict[key]['x'] * scale/image.shape[1])
        joint_dict[key]['y'] = int(joint_dict[key]['y'] * scale/image.shape[0])
    # plt.imsave(image_dir + "full_256\\" + new_data['image_name'], new_img)
    return data


def create_pad():
    padding = 256
    list_of_data = load_json(image_dir, 'mpii_singular_updated.json')
    list_of_imgs = os.listdir(image_dir + 'cut_images_256\\')
    for key in list_of_imgs:
        padded = pad_image(key, image_dir + "cut_images_256\\",
        padding, padding, image_dir + f"cut_images_{padding}_white\\")
        # if padded:
        #     train.display(image_dir + "padded_images\\", key, list_of_data[key])
        print(key)

    list_of_data = load_json(image_dir, 'mpii_256_shrunk.json')

    for key in list_of_data:
        if key not in os.listdir(image_dir + "cut_images_256\\"):
            print(0)


def find_change(new, old, triplet):
    new_data = new[[i for i in new.keys()][0]]['annorect']['annopoints']
    old_data = old[[i for i in old.keys()][0]]['annorect']['annopoints']
    # joint_ids = {
    #   '0':  'right ankle',
    #   '1':  'right knee',
    #   '2':  'right hip',
    #   '3':  'left hip',
    #   '4':  'left knee',
    #   '5':  'left ankle',
    #   '6':  'pelvis',
    #   '7':  'thorax',
    #   '8':  'upper neck',
    #   '9':  'head top',
    #   '10': 'right wrist',
    #   '11': 'right elbow',
    #   '12': 'right shoulder',
    #   '13': 'left shoulder',
    #   '14': 'left elbow',
    #   '15': 'left wrist'
    # }
    joint_ids = {
        'right_neck': '8',
        'left_neck': '8',
        'medial_right_shoulder': '12',
        'lateral_right_shoulder': '12',
        'medial_right_bow': '11',
        'lateral_right_bow': '11',
        'medial_right_wrist': '10',
        'lateral_right_wrist': '10',
        'medial_left_shoulder': '13',
        'lateral_left_shoulder': '13',
        'medial_left_bow': '14',
        'lateral_left_bow': '14',
        'medial_left_wrist': '15',
        'lateral_left_wrist': '15',
        'medial_right_hip': '2',
        'lateral_right_hip': '2',
        'medial_right_knee': '1',
        'lateral_right_knee': '1',
        'medial_right_ankle': '0',
        'lateral_right_ankle': '0',
        'medial_left_hip': '3',
        'lateral_left_hip': '3',
        'medial_left_knee': '4',
        'lateral_left_knee': '4',
        'medial_left_ankle': '5',
        'lateral_left_ankle': '5'
    }
    new_triplet = {}
    for key in triplet['contour_keypoints'].keys():
        id = joint_ids[key]
        new_triplet[id] = {}
        new_triplet[id]['x'] = triplet['contour_keypoints'][key][0] - (old_data[id]['x'] - new_data[id]['x'])
        new_triplet[id]['y'] = triplet['contour_keypoints'][key][1] - (old_data[id]['y'] - new_data[id]['y'])
    return new_triplet


image_dir = "F:\\Thesis Datasets\\mpii\\mpii_human_pose_v1\\"
# mat_file = "F:\\Thesis Datasets\\mpii_human_pose_v1_u12_2\\mpii_human_pose_v1_u12_1.mat"
# loaded_mat = loadmat(mat_file)
# cleanmat = clean_mat(loaded_mat)
# list_of_images = os.listdir(image_dir)
# image_set = train.create_dataset(image_dir, [[0, key] for key in cleanmat.keys()])
# create_json(cleanmat, image_dir, 'mpii_full.json')
list_of_data = load_json(image_dir, 'mpii_real.json')
triplet_list = load_json(image_dir, 'Triplet_Rep_Original.json')
old_data = load_json(image_dir, 'mpii_full.json')
# padding = 256
#
# new_list = copy.deepcopy(list_of_data)
# new_triplet = copy.deepcopy(triplet_data)
# for datapoint in triplet_data:
#     key = datapoint['img_name']
#     if key in os.listdir(f"{image_dir}cut_images\\"):
        # a = plt.imread(f"{image_dir}cut_images\\{datapoint['img_name']}")
        # plot.display_triplet(a, (datapoint, list_of_data[datapoint['img_name']]))
        # print(datapoint)
        # new_list[key] = full_image(plt.imread(os.path.join(image_dir + f"cut_images\\" + key)), list_of_data[key], datapoint)
        # _ = full_image(plt.imread(os.path.join(image_dir + f"cut_images\\" + key)), list_of_data[key], datapoint)
    # else:
    #     del(new_list[key])
    # rescale_image(key, 50, image_dir + 'cut_images\\', image_dir + 'cut_images_256\\')

# create_json(new_triplet, image_dir, 'Triplet_Rep_Full_Size.json')

triplet_data = {data['img_name']: data for data in triplet_list}
new_triplet = {}

for key in list_of_data.keys():
    if key  in old_data.keys():
        new_triplet[key] = find_change(list_of_data[key], old_data[key], triplet_data[key])

create_json(new_triplet, image_dir, 'Triplet_Rep_Full_Size.json')
