import json
import glob
import numpy as np
import pickle
import re
import os
import glob


PATH_TO_VIDEOS = '/home/ubuntu/ava_openpose'


def main():
    output = {
        'pose': [],
        'label': []
    }

    pattern = re.compile(r'^([a-z]+)\d+')

    for video in os.listdir(PATH_TO_VIDEOS):

        match = pattern.match(video)
        label = match.group(1)

        path_to_json_files = os.path.join(PATH_TO_VIDEOS, video, '*')
        openpose_file_paths = sorted(glob.glob(path_to_json_files))
        
        all_points = pose_from_openpose(openpose_file_paths)
        pose = np.array(all_points)

        output['label'].append(label)
        output['pose'].append(pose)

    pickle.dump(output, open('/home/ubuntu/ava_pose_label.pkl', 'wb'))


KEYPOINT_INDICES = [1, 2, 5, 9, 12, 3, 6, 10, 13, 4, 7, 11, 14]
    

def pose_from_openpose(file_paths):
    all_points = []
    for file_path in file_paths:
        points_for_frame = []
        with open(file_path) as json_file:
            content = json.load(json_file)
            people = content['people']

            if len(people) != 1:
                continue
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
