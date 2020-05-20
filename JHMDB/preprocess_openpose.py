import json
import scipy.io
import os
import glob
import re
import pickle
import numpy as np
from pathlib import Path
import fire


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


replace = 0
not_replace = 0


def main(
    split_dir= os.path.join(os.path.abspath(''), '..', 'data', 'JHMDB', 'GT_splits'),
    save_dir=os.path.join(os.path.abspath(''), '..', 'data', 'JHMDB_openpose_tracking_pkl'),
    mat_dir=os.path.join(os.path.abspath(''), '..', 'data', 'JHMDB', 'joint_positions'),
    openpose_dir=os.path.join(os.path.abspath(''), '..', 'data', 'JHMDB_openpose_tracking_json'),
    doctor=False

):

    GT_split_lists = list(Path(split_dir).rglob('*.txt')) 

    GT_lists_1 = []
    for filename in GT_split_lists:
        if (Path(filename).stem.split('_')[-1] == 'split1'):
            GT_lists_1.append(filename)

    train_set, test_set = generate_train_test_sets(GT_lists_1)
    
    train = {}
    train['pose'] = []
    train['label'] = []
    train['filename'] = []
    test = {}
    test['pose'] = []
    test['label'] = []
    test['filename'] = []

    good_count, bad_count = 0, 0
    
    for action_mat_dir in [x for x in Path(mat_dir).iterdir() if x.is_dir() and not x.stem.startswith('.')]:
        for vid_mat_dir in [y for y in action_mat_dir.iterdir() if y.is_dir() and not y.stem.startswith('.')]:
            action_name = action_mat_dir.stem
            vid_name = vid_mat_dir.stem
            mat = scipy.io.loadmat(str(vid_mat_dir / 'joint_positions.mat'))

            vid_op_dir = Path(openpose_dir) / (vid_mat_dir.relative_to(mat_dir))

            openpose_file_paths = sorted([str(x) for x in vid_op_dir.glob('*.json')])
            assert mat['pos_img'].shape[2] == len(openpose_file_paths)

            all_points = pose_from_openpose(openpose_file_paths, mat, doctor=doctor)
            if len(all_points) < 16:
                bad_count += 1
            else:
                good_count += 1
                # pose = np.array(generate_pose(mat))
                pose = np.array(all_points, dtype=np.float)

                if vid_name in train_set:
                    train['label'].append(action_name)
                    train['pose'].append(pose)
                    train['filename'].append(str(vid_mat_dir.relative_to(mat_dir)))
                elif vid_name in test_set:
                    test['label'].append(action_name)
                    test['pose'].append(pose)
                    test['filename'].append(str(vid_mat_dir.relative_to(mat_dir)))
                else:
                    raise Exception('Not in train or test')

    pickle.dump(train, open(os.path.join(save_dir, "GT_train_1.pkl"), "wb"))
    pickle.dump(test, open(os.path.join(save_dir, "GT_test_1.pkl"), "wb"))
    print('good count', good_count)
    print('bad count', bad_count)
    print(len(train['pose']), len(test['pose']))

    global replace
    global not_replace
    print('replace', replace, 'not_replace', not_replace)


JHMDB_KEYPOINT_INDICES = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    
def generate_pose(mat):
    '''Based on jhmdb_data_preprocessing_openpose'''

    points_all_frames = []
    for frame in range(len(mat['pos_img'][0][0])):
        points_for_frame = []

        # for joint in JHMDB_KEYPOINT_INDICES:
        for joint1 in JHMDB_KEYPOINT_INDICES:            
            x = mat['pos_img'][0][joint1 - 1][frame]
            y = mat['pos_img'][1][joint1 - 1][frame]

            point = [np.round(x, 3), np.round(y, 3)]
            points_for_frame.append(point)
        points_all_frames.append(points_for_frame)

    return points_all_frames


KEYPOINT_INDICES = [1, 2, 5, 9, 12, 3, 6, 10, 13, 4, 7, 11, 14]


OPENPOSE_TO_JHMDB = {
    1: 1,
    2: 4,
    5: 5,
    9: 6,
    12: 7,
    3: 8,
    6: 9,
    10: 10,
    13: 11,
    4: 12,
    7: 13,
    11: 14,
    14: 15,
}


def pose_from_openpose(file_paths, mat, doctor=False, score_thres=0.1):
    global replace
    global not_replace
    
    all_points = []
    for file_path, frame in zip(file_paths, range(len(mat['pos_img'][0][0]))):
        points_for_frame = []
        with open(file_path) as json_file:
            content = json.load(json_file)
            people = content['people']
            if len(people) == 0:
                continue

            max_iou = 0
            keypoints_max_iou = None
            for person in people:
                keypoints = people[0]['pose_keypoints_2d']
                if compute_iou(mat, frame, keypoints) > max_iou:
                    keypoints_max_iou = keypoints

            if keypoints_max_iou is None:
                continue

            keypoints = keypoints_max_iou

            for i in range(25):
                # for i, joint1 in zip(KEYPOINT_INDICES, JHMDB_KEYPOINT_INDICES):
                starting = i * 3
                x, y, score = keypoints[starting: starting+3]

                if score < score_thres:
                    x = y = 0.

                if doctor:
                    if x == 0 or y == 0:
                        replace += 1
                    else:
                        not_replace += 1

                    if i in OPENPOSE_TO_JHMDB:
                        jhmdb_index = OPENPOSE_TO_JHMDB[i]
                        if x == 0:
                            x = mat['pos_img'][0][jhmdb_index - 1][frame]
                        if y == 0:
                            y = mat['pos_img'][1][jhmdb_index - 1][frame]
                        
                # if x == 0:
                #     x = mat['pos_img'][0][joint1 - 1][frame]
                # if y == 0:
                #     y = mat['pos_img'][1][joint1 - 1][frame]

                point = [np.round(x, 3), np.round(y, 3)]
                points_for_frame.append(point)
            all_points.append(points_for_frame)

    return all_points

def compute_iou(mat, frame, openpose_keypoints):
    jhmdb_ymin = min(mat['pos_img'][1][joint][frame]
                     for joint in range(len(mat['pos_img'][1])))
    jhmdb_xmin = min(mat['pos_img'][0][joint][frame]
                     for joint in range(len(mat['pos_img'][0])))
    jhmdb_ymax = max(mat['pos_img'][1][joint][frame]
                     for joint in range(len(mat['pos_img'][1])))
    jhmdb_xmax = max(mat['pos_img'][0][joint][frame]
                     for joint in range(len(mat['pos_img'][0])))
    
    openpose_ymin = float('inf')
    openpose_xmin = float('inf')
    openpose_ymax = 0
    openpose_xmax = 0

    for i in KEYPOINT_INDICES:
        x = openpose_keypoints[i * 3]
        y = openpose_keypoints[(i * 3) + 1]
        
        if x > 0:
            if x < openpose_xmin:
                openpose_xmin = x
            if x > openpose_xmax:
                openpose_xmax = x
        if y > 0:
            if y < openpose_ymin:
                openpose_ymin = y
            if y > openpose_ymax:
                openpose_ymax = y

    jhmdb_box = [jhmdb_xmin, jhmdb_ymin, jhmdb_xmax, jhmdb_ymax]
    openpose_box = [openpose_xmin, openpose_ymin, openpose_xmax, openpose_ymax]

    return iou(jhmdb_box, openpose_box)

    
def iou(boxA, boxB):
    '''From https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/'''

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

    
if __name__ == '__main__':
    fire.Fire(main)
