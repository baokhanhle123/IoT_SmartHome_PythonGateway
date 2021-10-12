from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

import cv2
import imutils

import pyttsx3

audio = pyttsx3.init()
audio.setProperty('rate', 150)
audio.setProperty('volume', 0.6)

cam = cv2.VideoCapture(0)


def capture_image():
    frame = imutils.url_to_image('http://192.168.1.3:8888/out.jpg')
    cv2.imwrite('img_detect.png', frame)


isMask = False


def face_detection():
    # Disable scientific notation for clarity
    global isMask
    np.set_printoptions(suppress=True)
    # Load the model
    model = load_model('keras_model.h5')

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    image = Image.open('img_detect.png')
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)

    name = ["With Mask", "Without mask", "Back ground", "Wrong mask"]
    index = -1
    max_value = -1
    for i in range(0, len(prediction[0])):
        if max_value < prediction[0][i]:
            max_value = prediction[0][i]
            index = i
    print("Ket Qua : ", name[index])
    print("Chinh Xac: ", max_value)

    if index == 1 or index == 3:
        isMask = False
    elif index == 0:
        isMask = True
        audio.setProperty('volume', 0.6)

    if not isMask:
        audio.say("Wear your face mask")
        if audio.getProperty('volume') <= 0.9:
            audio.setProperty('volume', audio.getProperty('volume') + 0.1)
        elif audio.getProperty('volume') < 1.0:
            audio.setProperty('volume', 1.0)


while True:
    capture_image()
    face_detection()
    audio.runAndWait()
