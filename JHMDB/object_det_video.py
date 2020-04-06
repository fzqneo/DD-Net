import fire
import io
import itertools
import json
from multiprocessing import Pool
import pathlib
import random
import requests

def work_fn(p, video_dir, output_dir, detection_host):
    import cv2
    print("Trying to process ", p)
    cap = cv2.VideoCapture(str(p))
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fid = 0
    while True:
        # for each frame
        ret, frame = cap.read()
        if not ret:
            break
        # get detection results from server
        _, jpg_str = cv2.imencode('.jpg', frame)
        r = requests.post(
            'http://{}/detect'.format(random.choice(detection_host)),
            files={'image': io.BytesIO(jpg_str.tostring())}
        )
        assert r.ok

        data = r.json()
        data['origin_width'] = W
        data['origin_height'] = H

        # save to json
        q = output_dir / p.relative_to(video_dir).parent / p.stem / (p.stem + '_{:012d}_detection.json'.format(fid))
        q.parent.mkdir(parents=True, exist_ok=True)
        with q.open('wt') as f:
            json.dump(data, f)
        fid += 1

    cap.release()
    print("Processed {} frames from {} ".format(fid, p))
    return True
    

def main(
    video_dir='/home/zf/video-analytics/DD-Net/data/JHMDB_video/ReCompress_Videos',
    output_dir='/home/zf/video-analytics/DD-Net/data/JHMDB_coco_output',
    detection_host=['localhost:5000', 'localhost:5001']):

    if type(detection_host) not in (tuple, list):
        detection_host = [detection_host]

    video_dir = pathlib.Path(video_dir)
    output_dir = pathlib.Path(output_dir)
    assert video_dir.is_dir()
    assert output_dir.is_dir()

    path_list = list(video_dir.rglob('**/*.avi'))
    print("\n".join(list(map(str, path_list))))

    with Pool(4) as pool:
        pool.starmap(work_fn, list(zip(path_list, itertools.repeat(video_dir), itertools.repeat(output_dir), itertools.repeat(detection_host))))
        

if __name__ == "__main__":
    fire.Fire(main)