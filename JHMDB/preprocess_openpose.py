import json
import scipy.io
import os
import glob
import re
import pickle
import numpy as np


PATH_RE = '^/home/ubuntu/joint_positions/([A-Za-z_]+)/([a-zA-Z0-9_\-!@\(\)\+]+)$'


def read_annotation(annotation_path):
    lines = []
    with open(annotation_path) as f:
        lines.append(f.read().splitlines())
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
        if (filename.split('/')[-1].split('.')[0].split('_')[-1] == 'split1'):
            GT_lists_1.append(filename)

    train_set, test_set = generate_train_test_sets(GT_lists_1)
    
    train = {}
    train['pose'] = []
    train['label'] = []
    test = {}
    test['pose'] = []
    test['label'] = []

    good_count, bad_count = 0, 0
    
    prog = re.compile(PATH_RE)
    for mat_path in glob.glob('/home/ubuntu/joint_positions/*/*'):
        mat = scipy.io.loadmat(mat_path + '/joint_positions.mat')
        re_result = prog.match(mat_path)
        label = re_result.group(1)
        filename_without_avi = re_result.group(2)

        openpose_path = mat_path.replace(
            '/home/ubuntu/joint_positions', '/home/ubuntu/openface_jhmdb')

        openpose_file_paths = sorted(glob.glob(openpose_path + '/*'))
        assert mat['pos_img'].shape[2]==len(openpose_file_paths)

        bad = False
        for openpose_file_path in openpose_file_paths:
            with open(openpose_file_path) as json_file:
                json_content = json.load(json_file)
                people = json_content['people']
                if len(people) != 1:
                    bad = True

        if bad:
            bad_count += 1
        else:
            good_count += 1
            # pose = np.array(generate_pose(mat))
            pose = np.array(pose_from_openpose(openpose_file_paths))

            if filename_without_avi in train_set:
                train['label'].append(label)
                train['pose'].append(pose)
            elif filename_without_avi in test_set:
                test['label'].append(label)
                test['pose'].append(pose)
            else:
                raise Exception('Not in train or test')

    pickle.dump(train, open(os.path.join(save_dir, "GT_train_1.pkl"), "wb"))
    pickle.dump(test, open(os.path.join(save_dir, "GT_test_1.pkl"), "wb"))
    print('good count', good_count)
    print('bad count', bad_count)
    print(len(train['pose']), len(test['pose']))

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

    return points_all_frames


KEYPOINT_INDICES = [1, 8, 0, 2, 5, 9, 12, 3, 6, 10, 13, 4, 7, 11, 14]


def pose_from_openpose(file_paths):
    all_points = []
    for file_path in file_paths:
        points_for_frame = []
        with open(file_path) as json_file:
            content = json.load(json_file)
            people = content['people']
            if len(people) > 0:                
                keypoints = people[0]['pose_keypoints_2d']
                for i in range(25):
                    starting = i * 3
                    x = keypoints[starting]
                    y = keypoints[starting + 1]
                    point = [np.round(x, 3), np.round(y, 3)]
                    points_for_frame.append(point)
                all_points.append(points_for_frame)

    return all_points

if __name__ == '__main__':
    main()
