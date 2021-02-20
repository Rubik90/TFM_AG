import keras
import librosa
import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from config import EXAMPLES_PATH
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import moviepy.editor
import sys
import shutil

video = "./93.mp4"
vid = moviepy.editor.VideoFileClip(video)
audio = vid.audio
audio.write_audiofile("./" + "audio.wav")
class audioPredictions:

    def __init__(self, file):

        self.file = file
        self.path = './models/Affwild/audioModel.h5'
        self.loaded_model = keras.models.load_model(self.path)

    def make_predictions(self):

        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=1)
        x = np.expand_dims(x, axis=0)
        predictions = self.loaded_model.predict_classes(x)
        #print("Audio emotion prediction is", " ", self.convert_class_to_emotion(predictions))
        return self.convert_class_to_emotion(predictions)

    @staticmethod
    def convert_class_to_emotion(pred):

        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fear',
                            '6': 'disgusted',
                            '7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label

if __name__ == '__main__':
    # load model
    model = model_from_json(open("./models/Affwild/resnetAff.json", "r").read())
    # load weights
    model.load_weights('./models/Affwild/resnetAff.h5')
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture("93.mp4")
    chunks = os.listdir("./audioChunks/")
    i=0

    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter('predictions.avi', fourcc, 20,(frame_width,frame_height),True )

    font = cv2.FONT_HERSHEY_SIMPLEX 

    for chunk in chunks:

      live_prediction = audioPredictions(file="./audioChunks/" + chunk)

      ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
      if not ret:
          continue

      faces_detected = face_haar_cascade.detectMultiScale(test_img, 1.32, 5)
      visual_prediction = ""
      audio_prediction = ""

      for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        #roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        test_img = cv2.resize(test_img, (112, 112))
        img_pixels = image.img_to_array(test_img)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ("neutral","angry","disgusted","scared","happy","sad","surprised")
        visual_prediction = emotions[max_index]

        audio_prediction = live_prediction.make_predictions()

        print("Visual emotion prediction at second " + str(i) +" is :" + visual_prediction)
        print("Audio emotion prediction at second " + str(i) +" is :" + audio_prediction)

        if visual_prediction ==  audio_prediction:
          print("Multimodal emotion prediction at second " + str(i) +" is :" + audio_prediction)
        else:
          print("The singles modalities predictions do not match")

        resized_img = cv2.resize(test_img, (frame_width,frame_height))
        #cv2.putText(test_img, visual_prediction + audio_prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), cv2.LINE_4)
        
        cv2.putText(resized_img,  
                "Face is " + visual_prediction
                 + " and speech is " +audio_prediction,  
                (50, 50),  
                font, 1,  
                (0, 255, 255),  
                2,  
                2)

        out.write(resized_img)
      i=i+1

    cap.release()
    out.release()
    cv2.destroyAllWindows
