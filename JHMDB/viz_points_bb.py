import cv2
import json
import scipy.io
import glob


def main():
    mat_path = '/home/roger/Downloads/joint_positions/throw/Faith_Rewarded_throw_u_cm_np1_fr_med_33'

    cap_path = mat_path.replace(
        '/home/roger/Downloads/joint_positions',
        '/home/roger/Downloads/ReCompress_Videos')

    cap = cv2.VideoCapture(cap_path + '.avi')

    mat = scipy.io.loadmat(mat_path + '/joint_positions.mat')

    coco_path = mat_path.replace(
            '/home/roger/Downloads/joint_positions',
            '/home/roger/Downloads/JHMDB_coco_output')

    coco_file_paths = sorted(glob.glob(coco_path + '/*'))

    for frame_num in range(len(coco_file_paths)):
        coco_file_path = coco_file_paths[frame_num]
        _, frame_cv = cap.read()

        with open(coco_file_path) as json_file:
            json_content = json.load(json_file)

            for detection_box, detection_name, detection_score in zip(
                        json_content['detection_boxes'],
                        json_content['detection_names'],
                        json_content['detection_scores']):

                coco_ymin = (json_content['origin_height'] *
                             detection_box[0])
                coco_xmin = (json_content['origin_width'] *
                             detection_box[1])
                coco_ymax = (json_content['origin_height'] *
                             detection_box[2])
                coco_xmax = (json_content['origin_width'] *
                             detection_box[3])

                if detection_name == 'person' and detection_score > 0.5:
                    cv2.rectangle(
                        frame_cv, (int(coco_xmin), int(coco_ymin)),
                        (int(coco_xmax), int(coco_ymax)), (0, 0, 255), 1)
                    break

        jhmdb_ymin = min(mat['pos_img'][1][joint][frame_num]
                         for joint in range(len(mat['pos_img'][1])))
        jhmdb_xmin = min(mat['pos_img'][0][joint][frame_num]
                         for joint in range(len(mat['pos_img'][0])))
        jhmdb_ymax = max(mat['pos_img'][1][joint][frame_num]
                         for joint in range(len(mat['pos_img'][1])))
        jhmdb_xmax = max(mat['pos_img'][0][joint][frame_num]
                         for joint in range(len(mat['pos_img'][0])))

        if jhmdb_ymin < 0:
            jhmdb_ymin = 0
        if jhmdb_xmin < 0:
            jhmdb_xmin = 0
        if jhmdb_ymax > 240:
            jhmdb_ymax = 240
        if jhmdb_xmax > 320:
            jhmdb_xmax = 320

        cv2.rectangle(
            frame_cv, (int(jhmdb_xmin), int(jhmdb_ymin)),
            (int(jhmdb_xmax), int(jhmdb_ymax)), (0, 255, 0), 1)
        # for joint in range(len(mat['pos_img'][1])):
        #     cv2.circle(
        #         frame_cv,
        #         (int(mat['pos_img'][0][joint][frame_num]),
        #          int(mat['pos_img'][1][joint][frame_num])), 1,
        #         (0, 255, 0), -1)

        print(jhmdb_ymax, coco_ymax)
        print((jhmdb_ymin > coco_ymin or abs(jhmdb_ymin - coco_ymin) < 10)
                            , (jhmdb_xmin > coco_xmin or abs(jhmdb_xmin - coco_xmin) < 10)
                            , (jhmdb_ymax < coco_ymax or abs(jhmdb_ymax - coco_ymax) < 10)
                            , (jhmdb_xmax < coco_xmax or abs(jhmdb_xmax - coco_xmax) < 10))

        cv2.imshow('frame', frame_cv)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            return

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
