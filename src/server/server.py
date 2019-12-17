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
import keras.models
import base64
import sys
import os

BASE_DIR = '/Users/NavSha/Documents/tensorflow-projects/dogs-vs_cats'

sys.path.append(os.path.abspath("../preprocess"))
from load import *

#if not os.path.exists(os.path.join(BASE_DIR,'uploaded_images/')):
#    os.makedirs(os.path.join(BASE_DIR,'upload_images'))

UPLOAD_FOLDER = os.path.join(BASE_DIR,'upload_images')
ALLOWED_EXTENSIONS = set(['png','jpg','jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
global model, graph
model, graph = init()

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

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
            image_path = os.path.join(app.config['UPLOAD_FOLDER'])
            image.save(image_path)
            npimg = np.fromfile(image_path,np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            image = cv2.resize(image,(150,150))
            image = image.reshape(1,150,150,3)

            with graph.as_default():
                class_prediction = model.predict(image)
                if class_prediction == 1:
                    return jsonify(api_version = '0.1',
                                    id =1,
                                    model_name = "Dags vs cats classifier",
                                    name = filename,
                                    status = 200,
                                    type = 'Dog'), 200

                else:
                    return jsonify(api_version='0.1',
                                    id = 0,
                                    model_name = "Dogs vs cats classifier",
                                    name = filename,
                                    status = 200,
                                    type = 'Cat'), 400

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
