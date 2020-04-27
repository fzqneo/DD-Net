import json
import multiprocessing as mp
import pickle
import threading

import fire
import flask 
from logzero import logger
import numpy as np 

# Run Keras model in a separate process
def net_worker(job_q, res_q, model_path):
    import ddnet
    net = ddnet.load_DDNet(model_path)
    while True:
        X = job_q.get()
        out = net.predict(X)
        res_q.put(out)


# these are created in main()
le = None

# shared by processes, created in main
job_q = None
res_q = None
# used in the Flask process shared by request handling threads
q_lock = threading.Lock()

# cleaner for openpose
import ddnet
cleaner = ddnet.OpenPoseDataCleaner(copy=True, filter_joint_idx=ddnet.COMMON_GOOD_JOINTS_FROM_OP)

# config for ddnet
C = ddnet.DDNetConfig(frame_length=32, num_joints=len(cleaner.filter_joint_idx), joint_dim=2, num_classes=21, num_filters=32)

app = flask.Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def classify():
    if flask.request.method == 'POST':
        # if POST, run inference
        p = np.array(flask.request.get_json(force=True))
        # logger.debug(p)
        # OpenPose cleaning
        p_clean = cleaner.transform_point(p)
        # DDNet preprocessing (only on CPU)
        X_0, X_1 = ddnet.preprocess_batch([p_clean,], C)
        # DDNet inference (may run on GPU)
        with q_lock:
            job_q.put([X_0, X_1])
            proba = res_q.get()[0]

        rv = {
            'labels': le.classes_.tolist(),
            'scores': proba.tolist()
        }

        return flask.jsonify(rv)

    else:  
        # If GET, show the file upload form:
        return '''
        <!doctype html>
        <title>Action Classification</title>

        <h1>Use HTTP POST to post an array of openpose joints corresponding to a clip</h1>
        
        Python code example:

        <code style=display:block;white-space:pre-wrap>

        import numpy as np
        import json
        import requests

        arr = np.random.random((32, 25, 2)) # 32 frames, 25 joints, 2 dimensional. 32 can vary, but 25 and 2 must align with OpenPose.
        host = 'http://localhost:5000'  # change this
        r = requests.post(host, json=arr.tolist())
        assert r.ok

        result = r.json()
        print(json.dumps(result, indent=2))
        print("Predicted class: ", result['labels'][np.argmax(result['scores'])])

        </code>

        Output:

        <samp style=display:block;white-space:pre-wrap>
        {
        "labels": [
            "brush_hair",
            "catch",
            "clap",
            "climb_stairs",
            "golf",
            "jump",
            "kick_ball",
            "pick",
            "pour",
            "pullup",
            "push",
            "run",
            "shoot_ball",
            "shoot_bow",
            "shoot_gun",
            "sit",
            "stand",
            "swing_baseball",
            "throw",
            "walk",
            "wave"
        ],
        "scores": [
            0.0037275280337780714,
            0.018379023298621178,
            0.011386215686798096,
            0.017078900709748268,
            0.0007549828151240945,
            0.00517685292288661,
            0.5599587559700012,
            0.0018598985625430942,
            0.0037682815454900265,
            0.001879677176475525,
            0.00996250007301569,
            0.0019467660458758473,
            0.005872227717190981,
            0.010394759476184845,
            0.08965660631656647,
            0.002151297638192773,
            0.00022247909510042518,
            0.0012439972488209605,
            0.006871100049465895,
            0.009863458573818207,
            0.23784461617469788
        ]
        }
        Predicted class:  wave
        </samp>
        '''

def main(model_path= "../JHMDB/jhmdb_openpose_model_mixed_11.h5", le_path= "../JHMDB/jhmdb_le.pkl", port=5000):

    # Set up queues and start worker process
    global job_q
    global res_q
    job_q = mp.Queue()
    res_q = mp.Queue()

    worker_proc = mp.Process(target=net_worker, args=(job_q, res_q, model_path))
    worker_proc.daemon = True 
    worker_proc.start()

    global le
    with open(le_path, 'rb') as f:
        le = pickle.load(f)
        logger.info("Classes: " + str(le.classes_.tolist()))

    app.run(host='0.0.0.0',
            port=port,
            threaded=True)
    # Keras and TFlow doesn't work well with threading. If want threading, 
    # use multi processing and run the net in a separate process


if __name__ == '__main__':
    fire.Fire(main)

