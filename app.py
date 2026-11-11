from __future__ import division, print_function
#import sys
import os
import cv2
#import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import statistics as st
from keras.models import load_model
import time
from collections import Counter


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index1.html")
    
    
@app.route('/camera', methods = ['GET', 'POST'])
def camera():
    
    model=load_model("facial_expressions.h5")

    # Load the cascade
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)

    def extract_features(image):
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1)
        return feature / 255.0

    # Initialize the webcam
    webcam = cv2.VideoCapture(0)
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

    start_time = time.time()
    duration = 5  # Duration in seconds
    predictions = []  # Store predictions for final output

    while time.time() - start_time < duration:
        ret, im = webcam.read()
        if not ret:
            break
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (p, q, r, s) in faces:
            image = gray[q:q + s, p:p + r]
            cv2.rectangle(im, (p, q), (p + r, q + s), (0, 255, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]  # Get the full label from the dictionary
            predictions.append(prediction_label)  # Save for final output
            print("Predicted Output:", prediction_label)
            cv2.putText(im, prediction_label, (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        # cv2.imshow("Output", im)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):  # Press 'q' to quit
        #     break

    webcam.release()
    # cv2.destroyAllWindows()

    # Calculate the most common emotion
    if predictions:
        final_output1 = Counter(predictions).most_common(1)[0][0]
        print("Final Predicted Emotion:", final_output1)
        return render_template("buttons.html",final_output=final_output1)
    else:
        print("No predictions were made.")

@app.route('/templates/buttons', methods = ['GET','POST'])
def buttons():
    return render_template("buttons.html")
@app.route('/songs/surprise', methods = ['GET', 'POST'])
def songsSurprise():
    return render_template("songsSurprise.html")

@app.route('/songs/angry', methods = ['GET', 'POST'])
def songsAngry():
    return render_template("songsAngry.html")

@app.route('/songs/sad', methods = ['GET', 'POST'])
def songsSad():
    return render_template("songsSad.html")

@app.route('/songs/disgust', methods = ['GET', 'POST'])
def songsDisgust():
    return render_template("songsDisgust.html")

@app.route('/songs/happy', methods = ['GET', 'POST'])
def songsHappy():
    return render_template("songsHappy.html")

@app.route('/songs/fear', methods = ['GET', 'POST'])
def songsFear():
    return render_template("songsFear.html")

@app.route('/songs/neutral', methods = ['GET', 'POST'])
def songsNeutral():
    return render_template("songsNeutral.html")

@app.route('/templates/join_page', methods = ['GET', 'POST'])
def join():
    return render_template("join_page.html")
    
if __name__ == "__main__":
    app.run(debug=True)