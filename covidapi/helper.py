import tensorflow as tf
import cv2, json, os
from keras import layers
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = load_model(BASE_DIR+"/covidapi/model.h5")
pre_gen = ImageDataGenerator(rescale = 1.0/255)
gradModel = Model(inputs = [model.input], outputs = [model.layers[-21].output, model.output])


def getResult():
    prediction_gen = pre_gen.flow_from_directory(BASE_DIR + '/media/covid/')
    prediction = model.predict_generator(prediction_gen)[0]
    imageName = os.listdir(os.path.join(BASE_DIR, 'media', 'covid', 'imgdata'))[0]
    IMAGE_PATH = os.path.join(BASE_DIR, 'media', 'covid', 'imgdata', imageName)
    img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.backend.expand_dims(img, axis = 0)
    with tf.GradientTape() as tape:
        (convOutputs, predictions) = gradModel(img)
        loss = predictions[:, 0]
    output = convOutputs[0]
    grads = tape.gradient(loss, convOutputs)[0]
    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = np.ones(output.shape[0: 2], dtype = np.float32)
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    cam = cv2.resize(cam.numpy(), (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    img = np.array(img, dtype='uint8')
    img = img.reshape((224, 224, 3))
    output_image = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)
    heatPath = os.path.join(BASE_DIR, 'media', 'covid', 'imgdata', 'cam.png')
    cv2.imwrite(heatPath, output_image)
    return prediction