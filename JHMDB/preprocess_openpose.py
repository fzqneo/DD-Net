# import cv2
import scipy.io
import os
import glob
import re
import pickle
import numpy as np


PATH_RE = '^/home/ubuntu/joint_positions/([A-Za-z_]+)/([a-zA-Z0-9_\-!@\(\)\+]+)$'


def generate_train_test_list():
    GT_split_lists = glob.glob(os.path.join(
        os.path.join(os.path.abspath(''), '..', 'data', 'JHMDB'),
        'GT_splits/*.txt'))

    GT_lists_1 = []
    GT_lists_2 = []
    GT_lists_3 = []
    for file in GT_split_lists:
        if file.split('/')[-1].split('.')[0].split('_')[-1] == 'split1':
            GT_lists_1.append(file) 
        elif file.split('/')[-1].split('.')[0].split('_')[-1] == 'split2':
            GT_lists_2.append(file)
        elif file.split('/')[-1].split('.')[0].split('_')[-1] == 'split3':
            GT_lists_3.append(file)

    return GT_lists_1, GT_lists_2, GT_lists_3

def read_annotation(annotation_path):
    lines = []
    with open(annotation_path) as f:
        lines.append(f.read().splitlines() )
    f.close()
    #lines = np.sort(lines)
    lines = np.hstack(lines)
    return lines

def generate_train_test_sets(lists):
    train_set = set()
    test_set = set()
    for i in range(len(lists)):
        lines = read_annotation(lists[i])
        for line in lines:
            file_name, flag = line.split(' ')
            if flag == '1':
                train_set.add(file_name.split('.')[0])
            elif flag == '2':
                test_set.add(file_name.split('.')[0])
    return train_set, test_set

def main():
    data_dir = os.path.join(os.path.abspath(''), '..', 'data', 'JHMDB')
    save_dir = os.path.join(os.path.abspath(''), '..', 'data', 'JHMDB')

    GT_split_lists = glob.glob(os.path.join(data_dir, 'GT_splits/*.txt'))

    GT_lists_1 = []
    for filename in GT_split_lists:
        if filename.split('/')[-1].split('.')[0].split('_')[-1] == 'split1':
            GT_lists_1.append(filename)

    train_set, test_set = generate_train_test_sets(GT_lists_1)
    
    train = {}
    train['pose'] = []
    train['label'] = []
    test = {}
    test['pose'] = []
    test['label'] = []

    prog = re.compile(PATH_RE)
    
    for mat_path in glob.glob('/home/ubuntu/joint_positions/*/*'):
        mat = scipy.io.loadmat(mat_path + '/joint_positions.mat')
        re_result = prog.match(mat_path)
        label = re_result.group(1)
        filename_without_avi = re_result.group(2)

        openpose_path = mat_path.replace(
            '/home/ubuntu/joint_positions', '/home/ubuntu/openface_jhmdb')

        assert mat['pos_img'].shape[2]==len(glob.glob(openpose_path + '/*'))

        pose = np.array(generate_pose(mat))

        if filename_without_avi in train_set:
            train['label'].append(label)
            train['pose'].append(pose)
        elif filename_without_avi in test_set:
            test['label'].append(label)
            test['pose'].append(pose)

    pickle.dump(train, open(os.path.join(save_dir, "GT_train_1.pkl"), "wb"))
    pickle.dump(test, open(os.path.join(save_dir, "GT_test_1.pkl"), "wb"))


def scale_x(x, width, height, scale):
    '''Based on http://files.is.tue.mpg.de/jhmdb/README_joint_positions.txt'''

    return (((((float(x) / width) - 0.5) * width) / height) / scale)

def scale_y(y, height, scale):
    return (((float(y) / height) - 0.5) / scale)

def compute_x_scale(img_x, world_x, width, height):
    return (((((float(img_x) / width) - 0.5) * width) / height) / world_x)

def compute_y_scale(img_y, world_y, height):
    return (((float(img_y) / height) - 0.5) / world_y)

def verify_scale_back(mat, width, height):
    for frame in range(len(mat['pos_img'][0])):
        for joint in range(len(mat['pos_img'][0][0])):
            x_scale = compute_x_scale(
                mat['pos_img'][0][frame][joint],
                mat['pos_world'][0][frame][joint], width, height)
            y_scale = compute_y_scale(
                mat['pos_img'][1][frame][joint],
                mat['pos_world'][1][frame][joint], height)

            assert np.isclose(x_scale, y_scale)
            

def verify_scale(mat, width, height):
    for frame in range(len(mat['pos_img'][0])):
        scale = mat['scale'][0][frame]
        
        for joint in range(len(mat['pos_img'][0][0])):
            x = scale_x(mat['pos_img'][0][frame][joint], width, height, scale)
            
            # print('x', x, mat['pos_world'][0][frame][joint])
            if not np.isclose(x, mat['pos_world'][0][frame][joint]):
                print('x', x, mat['pos_world'][0][frame][joint])
                print(mat['pos_img'][0][frame][joint])

            y = scale_y(mat['pos_img'][1][frame][joint], height, scale)
            if not np.isclose(y, mat['pos_world'][1][frame][joint]):
                print('y', y, mat['pos_world'][1][frame][joint])
            # assert np.isclose(y, mat['pos_world'][1][frame][joint])

    return True

def generate_pose(mat):
    '''Based on jhmdb_data_preprocessing_openpose'''

    points_all_joints_all_frames = []
    for joint in range(len(mat['pos_img'][0][0])):    
        points_for_joint = []
        for frame in range(len(mat['pos_img'][0])):
            x = mat['pos_img'][0][frame][joint]
            y = mat['pos_img'][1][frame][joint]
            point = [np.round(x, 3), np.round(y, 3)]
            points_for_joint.append(point)
        points_all_joints_all_frames.append(points_for_joint)

    return points_all_joints_all_frames

if __name__ == '__main__':
    main()
