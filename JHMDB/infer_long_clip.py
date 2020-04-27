import json
import operator
import pathlib
import pickle

import fire
import numpy as np
import requests

# from openpose_new_data import pose_from_openpose

def pose_from_openpose(file_paths):
    all_points = []
    for file_path in file_paths:
        points_for_frame = []
        with open(file_path) as json_file:
            content = json.load(json_file)
            people = content['people']

            if len(people) != 1:
                continue
            assert len(people) == 1, "{} has more than 1 person".format(file_path)

            keypoints = people[0]['pose_keypoints_2d']

            for i in range(25):
                starting = i * 3
                x = keypoints[starting]
                y = keypoints[starting + 1]

                point = [np.round(x, 3), np.round(y, 3)]
                points_for_frame.append(point)
            all_points.append(points_for_frame)

    return all_points


def vote_majority(results):
    labels = results[0]['labels']
    top1 = [r['labels'][np.argmax(r['scores'])] for r in results]
    print("All labels", ",".join(top1))
    ordered_labels = sorted([(L, top1.count(L)) for L in labels], key=operator.itemgetter(1), reverse=True)
    print("Label counts: ", ordered_labels)

def vote_mean_score(results):
    labels = results[0]['labels']
    scores = np.array(list(map(operator.itemgetter('scores'), results)))
    print(scores.shape)
    mean_score = np.mean(scores, axis=0)
    ordered_labels = sorted(zip(labels, mean_score), key=operator.itemgetter(1), reverse=True)
    print("Label counts: ", ordered_labels)
    


def main(op_json_dir, span=32, stride=16, ddnet_host='http://localhost:5002'):
    all_json_list = sorted(map(str, pathlib.Path(op_json_dir).rglob("*.json")))

    X = np.array(pose_from_openpose(all_json_list))
    print(X.shape)        

    results = []
    for start in range(0, X.shape[0] - span, stride):
        p = X[start:start+span, :, :]
        r = requests.post(ddnet_host, json=p.tolist())
        assert r.ok
        result = r.json()
        results.append(result)

    print("\n---MAJORITY")
    vote_majority(results)
    print("\n---MEAN SCORE")
    vote_mean_score(results)


if __name__ == "__main__":
    fire.Fire(main)