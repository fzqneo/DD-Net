import json
import requests
import numpy as np

arr = np.random.random((32, 25, 2)) # 32 frames, 25 joints, 2 dimensional
r = requests.post('http://localhost:5000', json=arr.tolist())
assert r.ok
result = r.json()
print(json.dumps(result, indent=2))
print("Predicted class: ", result['labels'][np.argmax(result['scores'])])
