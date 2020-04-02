import cv2
import fire
import io
import json
import pathlib
import requests

def main(
    video_dir='/home/zf/video-analytics/DD-Net/data/JHMDB_video/ReCompress_Videos',
    output_dir='/home/zf/video-analytics/DD-Net/data/JHMDB_coco_output',
    detection_host='localhost:5000'):

    video_dir = pathlib.Path(video_dir)
    output_dir = pathlib.Path(output_dir)
    assert video_dir.is_dir()
    assert output_dir.is_dir()

    # for each video
    for p in video_dir.rglob('**/*.avi'):
        cap = cv2.VideoCapture(str(p))
        fid = 0
        while True:
            # for each frame
            ret, frame = cap.read()
            if not ret:
                break
            # get detection results from server
            _, jpg_str = cv2.imencode('.jpg', frame)
            r = requests.post(
                'http://{}/detect'.format(detection_host),
                files={'image': io.BytesIO(jpg_str.tostring())}
            )
            assert r.ok

            # save to json
            q = output_dir / p.relative_to(video_dir).parent / p.stem / (p.stem + '_{:012d}_detection.json'.format(fid))
            q.parent.mkdir(parents=True, exist_ok=True)
            with q.open('wt') as f:
                json.dump(r.json(), f)
            fid += 1

        cap.release()
        print("Processed ", p)


if __name__ == "__main__":
    fire.Fire(main)