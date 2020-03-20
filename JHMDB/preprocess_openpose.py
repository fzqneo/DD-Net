import cv2
import scipy.io
import os
import glob
import re
import numpy as np


PATH_RE = '^/home/ubuntu/joint_positions/([A-Za-z_]+)/[a-zA-Z0-9_\-!@\(\)\+]+$'


def generate_train_test_list():
    GT_split_lists = glob.glob(osp.join(C.data_dir, 'GT_splits/*.txt'))

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


def main():
    train = {}
    train['pose'] = []
    train['label'] = []
    test = {}
    test['pose'] = []
    test['label'] = []

    prog = re.compile(PATH_RE)
    
    for mat_path in glob.glob('/home/ubuntu/joint_positions/*/*'):
        mat = scipy.io.loadmat(mat_path + '/joint_positions.mat')
        print(mat_path)
        re_result = prog.match(mat_path)
        label = re_result.group(1)

        openpose_path = mat_path.replace(
            '/home/ubuntu/joint_positions', '/home/ubuntu/openface_jhmdb')

        assert mat['pos_img'].shape[2]==len(glob.glob(openpose_path + '/*'))

        video_prefix = mat_path.replace(
            '/home/ubuntu/joint_positions', '/home/ubuntu/ReCompress_Videos')

        # from https://stackoverflow.com/a/45416805/859277
        video = cv2.VideoCapture(video_prefix + '.avi')
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # pose = np.array(generate_pose(mat, width, height))
        
        print(mat_path)
        verify_scale(mat, width, height)
        
        # os.makedirs(output_path)

def scale_x(x, width, height, scale):
    '''Based on http://files.is.tue.mpg.de/jhmdb/README_joint_positions.txt'''

    return (((((float(x) / width) - 0.5) * width) / height) / scale)

def scale_y(y, height, scale):
    return (((float(y) / height) - 0.5) / scale)

def verify_scale(mat, width, height):
    for frame in range(len(mat['pos_img'][0])):
        scale = mat['scale'][0][frame]
        print(scale)
        
        for joint in range(len(mat['pos_img'][0][0])):
            x = scale_x(mat['pos_img'][0][frame][joint], width, height, scale)
            
            # print('x', x, mat['pos_world'][0][frame][joint])
            # assert np.isclose(x, mat['pos_world'][0][frame][joint])

            y = scale_y(mat['pos_img'][1][frame][joint], height, scale)
            if not np.isclose(y, mat['pos_world'][1][frame][joint]):
                print('y', y, mat['pos_world'][1][frame][joint])
            # assert np.isclose(y, mat['pos_world'][1][frame][joint])

    return True

def generate_pose(mat, width, height):
    '''Based on jhmdb_data_preprocessing_openpose'''

    joints = []
    for frame in range(len(mat['pos_img'][0])):
        scale = mat['scale'][0][frame]
        
        joints_for_frame = []
        for joint in range(len(mat['pos_img'][0][0])):
            x = scale_x(mat['pos_img'][0][frame][joint], width, height, scale)
            y = scale_y(mat['pos_img'][1][frame][joint], height, scale)
            joints_for_frame.append([np.round(x, 3), np.round(y, 3)])
        joints.append(joints_for_frame)

    return joints

if __name__ == '__main__':
    main()
