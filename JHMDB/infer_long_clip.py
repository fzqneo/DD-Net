import json
import operator
import pathlib
import pickle

import fire
import numpy as np
import requests
from tqdm import tqdm

# from openpose_new_data import pose_from_openpose

def pose_from_openpose(file_paths, score_thres=0.1):
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
                x, y, score = keypoints[starting: starting+3]

                if score < score_thres:
                    x = y = 0.

                point = [np.round(x, 3), np.round(y, 3)]
                points_for_frame.append(point)
            all_points.append(points_for_frame)

    return all_points


def vote_majority(results, score_thresh=0.5):
    """[summary]

    Arguments:
        results {list of dict} -- Each dict is a prediction from the web API
        score_thres {float} -- Only score higher than this will be counted.
    
    Returns: list of (cls_name, score) ordered by score
    """
    labels = results[0]['labels']
    top1 = [r['labels'][np.argmax(r['scores'])] for r in results if np.max(r['scores']) >= score_thresh]
    print("Short predictions:", ",".join(top1))
    ordered_labels = sorted([(L, top1.count(L)) for L in labels], key=operator.itemgetter(1), reverse=True)
    if ordered_labels:
        return ordered_labels
    else:
        return ['None', 0.]

def vote_mean_score(results):
    labels = results[0]['labels']
    scores = np.array(list(map(operator.itemgetter('scores'), results)))
    # print(scores.shape)
    mean_score = np.mean(scores, axis=0)
    ordered_labels = sorted(zip(labels, mean_score), key=operator.itemgetter(1), reverse=True)
    return ordered_labels
    

def multi_predict_clip(clip_json_dir, span=32, stride=16, ddnet_host='http://localhost:5000'):
    all_json_list = sorted(map(str, pathlib.Path(clip_json_dir).rglob("*.json")))

    X = np.array(pose_from_openpose(all_json_list))
    print(X.shape)        

    results = []
    for start in range(0, X.shape[0], stride):
        p = X[start:start+span, :, :]
        r = requests.post(ddnet_host, json=p.tolist())
        assert r.ok
        result = r.json()
        results.append(result)

    return results


def infer_one_clip(clip_json_dir, voting='majority', *args, **kwargs):
    results = multi_predict_clip(clip_json_dir, *args, **kwargs)
    if voting == 'majority':
        return vote_majority(results)
    elif voting == 'mean':
        return vote_mean_score(results)
    else:
        raise ValueError(voting)


def eval_one_class(top_dir, target_class, *args, **kwargs):
    total = 0
    correct = 0
    for clip_json_dir in tqdm([p for p in pathlib.Path(top_dir).glob("*") if p.is_dir()]):
        print(clip_json_dir.name)
        total += 1
        prediction = infer_one_clip(clip_json_dir, *args, **kwargs)
        pred_cls, pred_score = prediction[0]
        correct += int(pred_cls == target_class)
        print(clip_json_dir.name, pred_cls, pred_score, pred_cls==target_class)

    print("Total", total, "Correct", correct, "Accuracy", correct/total)


def eval_one_class_exist(top_dir, target_class, *args, **kwargs):
    total = 0
    exist = 0
    for clip_json_dir in tqdm([p for p in pathlib.Path(top_dir).glob("*") if p.is_dir()]):
        total += 1
        results = multi_predict_clip(clip_json_dir)
        ordered_labels = vote_majority(results)
        ordered_labels = dict(ordered_labels)
        good = ordered_labels.get(target_class, 0) > 0
        exist += int(good)
        print(clip_json_dir.name, good)

    print("Total", total, "Exist", exist, "Exist rate", exist/total)        



if __name__ == "__main__":
    fire.Fire()