import sys
try:
    import numpy.core as _c; sys.modules['numpy._core'] = _c; sys.modules['numpy._core.multiarray'] = _c.multiarray
except: pass

from flask import Flask, request, jsonify
import numpy as np
from stable_baselines3 import PPO

MODEL = PPO.load('ppo_hft_scalper.zip')  # Load model

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    w = np.array(request.json['window'], dtype=np.float32)
    if w.shape != (50, 5):
        return jsonify(error='50x5 required'), 400
    a, _ = MODEL.predict(w, deterministic=True)
    sig = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}[int(a)]
    return jsonify(action=int(a), signal=sig)

if __name__ == '__main__':
    # FIXED HOST AND PORT
    app.run(host='0.0.0.0', port=10000)
