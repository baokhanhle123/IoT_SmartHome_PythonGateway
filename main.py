from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

import cv2
import imutils

import pyttsx3

import smtplib
import imghdr
from email.message import EmailMessage

# Image server link
imgLink = 'http://smarthomecamera.ddns.net:8888/out.jpg'

# Want to receive email?
isSendEmail = True

EMAIL_ADDRESS = "bksmartiot@gmail.com"
EMAIL_PASSWORD = "abcd@12345"

contacts = ['baokhanhle.278@gmail.com', 'test@example.com']

msg = EmailMessage()
msg['Subject'] = 'Abnormal behavior detected!'
msg['From'] = EMAIL_ADDRESS
msg['To'] = 'baokhanhle.278@gmail.com'

msg.set_content('Suspicious person detected!')


def capture_image_email():
    print("Capture image 1...")
    frame = imutils.url_to_image(imgLink)
    cv2.imwrite('img_email1.png', frame)

    print("Capture image 2...")
    frame = imutils.url_to_image(imgLink)
    cv2.imwrite('img_email2.png', frame)

    #print("Capture image 3...")
    #frame = imutils.url_to_image(imgLink)
    #cv2.imwrite('img_email3.png', frame)


def send_email():
    capture_image_email()
    files = ['img_email1.png', 'img_email2.png']
    #files = ['img_email1.png', 'img_email2.png', 'img_email3.png']

    for file in files:
        with open(file, 'rb') as f:
            file_data = f.read()
            file_type = imghdr.what(f.name)
            file_name = f.name

        msg.add_attachment(file_data, maintype='image', subtype=file_type, filename=file_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        print("Logged in...")
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        print("Sending email...")
        smtp.send_message(msg)
        print("Email has been sent!")


# Setting for audio
audio = pyttsx3.init()
audio.setProperty('rate', 150)
audio.setProperty('volume', 0.6)

# Setting for image capture
cam = cv2.VideoCapture(0)


# Capture image from server
def capture_image():
    frame = imutils.url_to_image(imgLink)
    cv2.imwrite('img_detect.png', frame)


isSended = False


# Face mask detection
def check_face_mask(index):
    isMask = False
    global isSended

    if index == 1 or index == 3:
        isMask = False
    elif index == 0:
        isMask = True
        isSended = False
        audio.setProperty('volume', 0.6)

    if not isMask:
        if isSendEmail and not isSended:
            send_email()
            print("Email sended!")
            isSended = True

        audio.say("Wear your face mask")
        if audio.getProperty('volume') <= 0.9:
            audio.setProperty('volume', audio.getProperty('volume') + 0.1)
        elif audio.getProperty('volume') < 1.0:
            audio.setProperty('volume', 1.0)


# Face detection
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

    isSended = False
    check_face_mask(index)

while True:
    capture_image()
    face_detection()
    audio.runAndWait()