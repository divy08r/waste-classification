from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

MODEL_PATH = 'network1.h5'

model = load_model(MODEL_PATH)
# model._make_predict_function()          

print('Model loaded. Check http://127.0.0.1:5000/')


# def model_predict(img_path, model):
#     img = image.load_img(img_path, target_size=(224, 224))

#     x = image.img_to_array(img)

#     x = np.expand_dims(x, axis=0)

#     x = preprocess_input(x, mode='caffe')

#     preds = model.predict(x)
#     return preds

from tensorflow.keras.preprocessing import image
output_class = ["battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass", "metal", "paper", "plastic","shoes","trash","white-glass"]
def model_predict(new_image):
  t_img = image.load_img(new_image, target_size = (256,256))

  t_img = image.img_to_array(t_img)/255
  t_img = np.expand_dims(t_img, axis=0)

  predicted_array = model.predict(t_img)
  predicted_value = output_class[np.argmax(predicted_array)]
  predicted_accuracy = round(np.max(predicted_array) * 100, 2)

  print("Your waste material is ", predicted_value, " with ", predicted_accuracy, " % accuracy")
  return predicted_value


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)


        preds = model_predict(file_path)

        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=3000, debug=True)