import scipy.io as spio
import os, json
import numpy as np
import copy
import matplotlib.pyplot as plt

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def is_struct(struct_obj):
    '''
    Checks if the object given is a matlab struct
    '''
    if isinstance(struct_obj, spio.matlab.mio5_params.mat_struct):
        return True
    else:
        return False


def make_annopoints(joints):
    '''
    Creates a dict of joints from the struct provided
    '''
    joint_ids = {
      '0':  'right ankle',
      '1':  'right knee',
      '2':  'right hip',
      '3':  'left hip',
      '4':  'left knee',
      '5':  'left ankle',
      '6':  'pelvis',
      '7':  'thorax',
      '8':  'upper neck',
      '9':  'head top',
      '10': 'right wrist',
      '11': 'right elbow',
      '12': 'right shoulder',
      '13': 'left shoulder',
      '14': 'left elbow',
      '15': 'left wrist'
    }
    return {
        str(joint.id): {
            'x':            joint.x,
            'y':            joint.y,
            'name':         joint_ids[str(joint.id)],
            'is_visible':   joint.is_visible
        }
        for joint in joints
    }


def has_joints(struct_obj, required_joints=None):
    '''
    Checks if the struct has all the joints required.
    '''
    if required_joints is None:
        required_joints = list(range(0, 16))
    nums = []
    for item in struct_obj:
        nums.append(item.id)
    if all(joint in nums for joint in required_joints):
        return True
    else:
        return False


def make_dict(image_struct, annorect, mat, i):
    '''
    Creates a dict out of the matlab struct provided
    '''
    return {
        'image_name':       image_struct.image.name,
        'annorect':         {
            'x1':               annorect.x1,
            'y1':               annorect.y1,
            'x2':               annorect.x2,
            'y2':               annorect.y2,
            'scale':            annorect.scale,
            'objpos':           {
                'x': annorect.objpos.x,
                'y': annorect.objpos.y
            },
            'annopoints':       make_annopoints(annorect.annopoints.point),
                             },
        'img_train':        mat['RELEASE']['img_train'][i],
        'single_person':    mat['RELEASE']['single_person'][i],
        'act': {
            'act_id':       mat['RELEASE']['act'][i].act_id,
            'act_name':     mat['RELEASE']['act'][i].act_name,
            'cat_name':     mat['RELEASE']['act'][i].cat_name
                },
        'video_list':       mat['RELEASE']['video_list'][image_struct.vididx-1]
        if isinstance(image_struct.vididx, int) else None
    }


def is_valid(annorect):
    '''
    Checks if the annorect provided has the right fields
    '''
    annorect_items = ['x1', 'y1', 'x2', 'y2', 'scale', 'objpos', 'annopoints']
    if all(item in annorect._fieldnames for item in annorect_items):
        if hasattr(annorect.annopoints, '_fieldnames'):
            if not is_struct(annorect.annopoints.point) and has_joints(annorect.annopoints.point):
                return True
    return False


def clean_mat(mat):
    clean_dict = {}
    for i in range(len(mat['RELEASE']['annolist'])):
        image_struct = mat['RELEASE']['annolist'][i]
        if is_struct(image_struct):
            if is_struct(image_struct.annorect):
                #Struct of annorect
                if is_valid(image_struct.annorect):
                    clean_dict[image_struct.image.name] = {str(i): make_dict(image_struct, image_struct.annorect, mat, i)}
            else:
                pass
                #Array of annorect structs, need to grab individual ones
                # for j in range(len(image_struct.annorect)):
                #     if is_valid(image_struct.annorect[j]):
                #         clean_dict[image_struct.image.name] = {f'{i},{j}': make_dict(image_struct, image_struct.annorect[j], mat, i)}
        else:
            for anno in image_struct.annorect:
                if "x1" in anno:
                    pass  # continue
                else:
                    pass  # remove
    return clean_dict


def to_int(num_array):
    new_array = [None] * len(num_array)
    for item in range(len(num_array)):
        if isinstance(num_array[item], np.uint8):
            new_array[item] = int(item)
        elif isinstance(num_array[item], list):
            new_array[item] = to_int(num_array[item])
        else:
            new_array[item] = int
    return new_array


def to_list(dic):
    list_dict = {}
    for key in dic.keys():
        if isinstance(dic[key], dict):
            list_dict[key] = to_list(dic[key])
        elif isinstance(dic[key], np.ndarray):
            list_dict[key] = to_int(dic[key].tolist())
        elif isinstance(dic[key], np.uint8):
            list_dict[key] = int(dic[key])
        else:
            list_dict[key] = dic[key]
    return list_dict


def create_json(dic, directory, name):
    with open(os.path.join(directory, name), 'w+') as infile:
        json.dump(to_list(dic), infile)


def load_json(directory, name):
    with open(os.path.join(directory, name), 'r') as infile:
        return json.load(infile)


def reverse_dic(dic, directory):
    new_dic = copy.deepcopy(dic)
    for image in dic.keys():
        new_name = image.strip('.jpg')[::-1] + '.jpg'
        img = plt.imread(directory + image)
        if new_name == image:
            new_name = new_name.strip('.jpg') + '-1.jpg'
        new_dic[new_name] = new_dic[image]
        del(new_dic[image])
        for person in dic[image]:
            new_dic[new_name][person]['image_name'] = new_name
            annorect = new_dic[new_name][person]['annorect']
            new_dic[new_name][person]['annorect'] = update_loc(annorect, img.shape[0], img.shape[1])
    dic.update(new_dic)
    # return new_dic


def update_loc(annorect, y, x):
    if x > 0:
        annorect['x1'] = x - annorect['x1']
        annorect['x2'] = x - annorect['x2']
    if y > 0:
        annorect['y1'] = y - annorect['y1']
        annorect['y2'] = y - annorect['y2']
    for joint in annorect['annopoints'].keys():
        if x > 0:
            annorect['annopoints'][joint]['x'] = x - annorect['annopoints'][joint]['x']
        if y > 0:
            annorect['annopoints'][joint]['y'] = y - annorect['annopoints'][joint]['y']
    return annorect
