import json
import os
import scipy.io
import glob


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



def main():
    good = 0
    bad = 0
    no_prediction = 0
    yes_prediction = 0

    data_dir = os.path.join(os.path.abspath(''), '..', 'data', 'JHMDB')
    for mat_path in glob.glob('/home/roger/Downloads/joint_positions/*/*'):
        mat = scipy.io.loadmat(mat_path + '/joint_positions.mat')

        coco_path = mat_path.replace(
            '/home/roger/Downloads/joint_positions',
            '/home/roger/Downloads/JHMDB_coco_output')

        coco_file_paths = sorted(glob.glob(coco_path + '/*'))
        assert mat['pos_img'].shape[2]==len(coco_file_paths)

        for frame in range(len(coco_file_paths)):
            coco_file_path = coco_file_paths[frame]

            person_prediction = False
            found_correct_person = False
            with open(coco_file_path) as json_file:
                json_content = json.load(json_file)

                for detection_box, detection_name, detection_score in zip(
                        json_content['detection_boxes'],
                        json_content['detection_names'],
                        json_content['detection_scores']):
                    if detection_name == 'person' and detection_score > 0.5:
                        person_prediction = True

                        assert json_content['origin_height'] == 240
                        assert json_content['origin_width'] == 320

                        coco_ymin = (json_content['origin_height'] *
                                     detection_box[0])
                        coco_xmin = (json_content['origin_width'] *
                                     detection_box[1])
                        coco_ymax = (json_content['origin_height'] *
                                     detection_box[2])
                        coco_xmax = (json_content['origin_width'] *
                                     detection_box[3])

                        jhmdb_ymin = min(mat['pos_img'][1][joint][frame]
                                         for joint in range(len(mat['pos_img'][1])))
                        jhmdb_xmin = min(mat['pos_img'][0][joint][frame]
                                         for joint in range(len(mat['pos_img'][0])))
                        jhmdb_ymax = max(mat['pos_img'][1][joint][frame]
                                         for joint in range(len(mat['pos_img'][1])))
                        jhmdb_xmax = max(mat['pos_img'][0][joint][frame]
                                         for joint in range(len(mat['pos_img'][0])))

                        if jhmdb_ymin < 0:
                            jhmdb_ymin = 0
                        if jhmdb_xmin < 0:
                            jhmdb_xmin = 0
                        if jhmdb_ymax > 240:
                            jhmdb_ymax = 240
                        if jhmdb_xmax > 320:
                            jhmdb_xmax = 320

                        coco_box = [coco_xmin, coco_ymin, coco_xmax, coco_ymax]
                        jhmdb_box = [jhmdb_xmin, jhmdb_ymin, jhmdb_xmax, jhmdb_ymax]

                        if (((jhmdb_ymin > coco_ymin or abs(jhmdb_ymin - coco_ymin) < 10)
                            and (jhmdb_xmin > coco_xmin or abs(jhmdb_xmin - coco_xmin) < 10)
                            and (jhmdb_ymax < coco_ymax or abs(jhmdb_ymax - coco_ymax) < 10)
                            and (jhmdb_xmax < coco_xmax or abs(jhmdb_xmax - coco_xmax) < 10)) or
                        iou(coco_box, jhmdb_box) > 0.5):
                            found_correct_person = True

                if found_correct_person:
                    good += 1
                else:
                    bad += 1

                if person_prediction:
                    yes_prediction += 1
                else:
                    no_prediction += 1

                if person_prediction and not found_correct_person:
                    print(mat_path, frame)

    print('good', good, 'bad', bad)
    print('prediction: yes', yes_prediction, 'no', no_prediction)


if __name__ == '__main__':
    main()
