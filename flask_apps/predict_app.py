from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import base64
import io
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

from src.config import MODEL_DIR
from flask_apps.server_variables import UPLOAD_DIR

app = Flask(__name__)

model_dir = MODEL_DIR / 'new_architecture'
image_dir = UPLOAD_DIR

def get_model(path):
    global model
    model = load_model(path)
    print(" * Model loaded!")


def preprocess_image_cnn(path):
    '''preprocess image'''
    img_size = 128
    # get latest uploaded file
    list_of_files = glob.glob(path + '/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f' latest file {latest_file}')

    img_array = cv2.imread(latest_file)
    img_array = rgb2gray(img_array)  # transform to grayscale image
    img_array = cv2.GaussianBlur(img_array, (5, 5), cv2.BORDER_DEFAULT)
    img_array = cv2.resize(img_array, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

    return img_array


def preprocess_image(image, target_size):
    print(image.format, image.size, image.mode)
    print('start processing')
    print(np.shape(image))

    if image.mode != 'L':
        # image = rgb2gray(image)
        image = image.convert('L')

    print(np.shape(image))
    image = img_to_array(image)
    print(np.shape(image))
    print(type(image))
    if np.shape(image) == (None, None, 3):
        print('rgb2gray')
        image = rgb2gray(image)
    else:
        print('nothing happens')
        pass
    # image = rgb2gray(image)
    # print(np.shape(image))
    # if image.mode != 'L':
    #     # image = rgb2gray(image)
    #     image = image.convert('L')
    # # image = image.resize(target_size)
    image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
    print(f' nach blur {np.shape(image)}')
    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    # plt.imshow(image)
    plt.show()
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    print(np.shape(image))

    return image


print('Loading Keras Model')
get_model(path=(str(model_dir) + '\cnn_model.h5'))


@app.route('/predict', methods=['Post'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=128)
    print('prediction starts now')
    prediction = model.predict(processed_image)
    prob_pneumenic = prediction.tolist()
    prob_normal = (1 - prediction).tolist()
    print('prediction finished')
    response = {
        'prediction': {
            'normal': prob_normal,
            'pneumenic': prob_pneumenic
        }
    }
    print(response)
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
