'''
Summary:
  Flask app that exposes an API

Returns:
  Returns a JSON stating whether an image is of a dog or a cat.
'''

from flask import Flask, render_template, request,jsonify
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import tensorflow.keras.models
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

sys.path.append(os.path.join(SCRIPT_DIR, "../preprocess"))
from load import *

UPLOAD_FOLDER = os.path.join(PROJECT_DIR,'upload_images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = set(['png','jpg','jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
global model
model = init()

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/class_pred", methods = ["GET","POST"])
def class_predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify(api_version='0.1',
                            model_name = "Dogs vs cats classifier",
                            error = 'Upload a file',
                            status = 400), 400
        image = request.files['image']
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            npimg = np.fromfile(image_path,np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            image = cv2.resize(image,(150,150))
            image = image.reshape(1,150,150,3) / 255.0

            score = float(model.predict(image)[0][0])
            is_dog = score >= 0.5
            confidence = score if is_dog else 1 - score
            return jsonify(api_version = '0.1',
                            id = 1 if is_dog else 0,
                            model_name = "Dogs vs cats classifier",
                            name = filename,
                            status = 200,
                            type = 'Dog' if is_dog else 'Cat',
                            confidence = round(confidence, 4)), 200

        else:
            return jsonify(api_version='0.1',
                            model_name = "Dogs vs cats classifier",
                            error = 'Upload a valid file',
                            status = 400), 400
    else:
        return  jsonify(api_version='0.1',
        model_name = "Dogs vs cats classifier",
        error = 'Only accepts POST method',
        status = 405), 405

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host = '0.0.0.0', port=port,debug=False)
