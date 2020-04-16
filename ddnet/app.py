import json
import pickle

import fire
import flask 
from logzero import logger
import numpy as np 

import ddnet

# these are created in main()
net = None
le = None


# cleaner for openpose
cleaner = ddnet.OpenPoseDataCleaner(copy=False)

# config for ddnet
C = ddnet.DDNetConfig(frame_length=32, num_joints=len(cleaner.filter_joint_idx), joint_dim=2, num_classes=21, num_filters=32)

app = flask.Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def classify():
    if flask.request.method == 'POST':
        # run inference
        p = np.array(flask.request.get_json(force=True))
        # logger.debug(p)
        p_clean = cleaner.transform_point(p)
        X_0, X_1 = ddnet.preprocess_batch([p_clean,], C)
        proba = net.predict([X_0, X_1])[0]

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

        arr = np.random.random((32, 25, 2)) # 32 frames, 25 joints, 2 dimensional
        r = requests.post('http://localhost:5000', json=arr.tolist())
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

def main(model_path= "../JHMDB/jhmdb_openpose_model.h5", le_path= "../JHMDB/jhmdb_le.pkl"):
    global le
    global net
        
    net = ddnet.load_DDNet(model_path)
    with open(le_path, 'rb') as f:
        le = pickle.load(f)
        logger.info("Classes: ", le.classes_.tolist())

    app.run(host='0.0.0.0',
            port=5000,
            threaded=False)


if __name__ == '__main__':
    fire.Fire(main)

