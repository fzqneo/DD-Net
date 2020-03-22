# import cv2
import json
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
    save_dir = os.path.join(os.path.abspath(''), '..', 'data', 'JHMDB_openpose')

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

        openpose_file_paths = glob.glob(openpose_path + '/*')
        assert mat['pos_img'].shape[2]==len(openpose_file_paths)

        # pose = np.array(generate_pose(mat))
        num_keypoints = get_num_keypoints(openpose_file_paths)
        if num_keypoints is None:
            print('None', mat_path)
        else:
            pose = np.array(pose_from_openpose(openpose_file_paths, num_keypoints))

            if filename_without_avi in train_set:
                train['label'].append(label)
                train['pose'].append(pose)
            elif filename_without_avi in test_set:
                test['label'].append(label)
                test['pose'].append(pose)

    pickle.dump(train, open(os.path.join(save_dir, "GT_train_1.pkl"), "wb"))
    pickle.dump(test, open(os.path.join(save_dir, "GT_test_1.pkl"), "wb"))

def generate_pose(mat):
    '''Based on jhmdb_data_preprocessing_openpose'''

    points_all_frames = []
    for frame in range(len(mat['pos_img'][0][0])):
        points_for_frame = []

        for joint in range(len(mat['pos_img'][0])):
            x = mat['pos_img'][0][joint][frame]
            y = mat['pos_img'][1][joint][frame]
            point = [np.round(x, 3), np.round(y, 3)]
            points_for_frame.append(point)
        points_all_frames.append(points_for_frame)

    return points_all_joints_all_frames

def find_person_most_points(people):
    most_nonzeros = 0
    index_most_nonzeros = 0
    for i in range(len(people)):
        person = people[i]
        nonzeros = 0
        for point in people[i]['pose_keypoints_2d']:
            if point > 0:
                nonzeros += 1

        if nonzeros > most_nonzeros:
            most_nonzeros = nonzeros
            index_most_nonzeros = i

    return index_most_nonzeros

def get_num_keypoints(file_paths):
    for file_path in file_paths:
        with open(file_path) as json_file:
            json_content = json.load(json_file)
            people = json_content['people']
            if len(people) > 0:
                num_keypoints = len(people[0]['pose_keypoints_2d'])
                assert (num_keypoints % 3) == 0
                return num_keypoints

    return None

def pose_from_openpose(file_paths, num_keypoints):
    all_points = []
    for file_path in file_paths:
        points_for_frame = []
        with open(file_path) as json_file:
            content = json.load(json_file)
            people = content['people']
            if len(people) > 0:
                for i in range(0, num_keypoints, 3):
                    person_index = find_person_most_points(people)

                    keypoints = people[person_index]['pose_keypoints_2d']
                    x = keypoints[i]
                    y = keypoints[i + 1]
                    point = [np.round(x, 3), np.round(y, 3)]
                    points_for_frame.append(point)
                all_points.append(points_for_frame)

    return all_points

if __name__ == '__main__':
    main()
