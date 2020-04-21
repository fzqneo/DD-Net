import json
import glob
import numpy as np
import pickle


ACTIONS = ['clap1', 'clap2', 'wave1', 'wave2']
LABELS = ['clap', 'clap', 'wave', 'wave']


def main():
    output = {
        'pose': [],
        'label': []
    }


    for action, label in zip(ACTIONS, LABELS):
        openpose_path = '/home/ubuntu/roger_actions_output/' + action
        openpose_file_paths = sorted(glob.glob(openpose_path + '/*'))
        
        all_points = pose_from_openpose(openpose_file_paths)
        pose = np.array(all_points)

        output['label'].append(label)
        output['pose'].append(pose)

    pickle.dump(output, open('/home/ubuntu/roger_actions_output/pose_label.pkl', 'wb'))


KEYPOINT_INDICES = [1, 2, 5, 9, 12, 3, 6, 10, 13, 4, 7, 11, 14]
    

def pose_from_openpose(file_paths):
    all_points = []
    for file_path in file_paths:
        points_for_frame = []
        with open(file_path) as json_file:
            content = json.load(json_file)
            people = content['people']

            assert len(people) == 1

            keypoints = people[0]['pose_keypoints_2d']

            # for i in KEYPOINT_INDICES:
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
