
import os
import io
import json
import time
import joblib
import numpy as np
import tensorflow as tf

from PIL import Image
from flask import Flask, request, jsonify

# load model
t0 = time.time()
model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'model.h5'
    )
model = tf.keras.models.load_model(model_path)
print('**Model loaded in {:0.2f} seconds.'.format(time.time() - t0))

# load class_map
class_map_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'class_map.pkl'
    )    
class_map = joblib.load(class_map_path)
class_map = {v:k for k, v in class_map.items()}

# crear la API
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # read image    
    data = json.loads(request.data)['instances']
    data = np.asarray(data)    
    # prediction
    prediction = model.predict(data)
    # retornar json
    return jsonify({'predictions': prediction.tolist()})

app.run()


