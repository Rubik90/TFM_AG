from os import listdir
from PIL import Image
import os
import cv2

dataset = "./cleaned/"

train_path = dataset + "train_frames/"
val_path = dataset + "val_frames/"
test_path = dataset + "test_frames/"
sets = [train_path, val_path, test_path]

i=0
for dataset in sets:
  emotions = os.listdir(dataset)
  for emotion in emotions:
    frames = os.listdir(train_path+emotion)
    for frame in frames:
      if frame.endswith('.bmp'):
        img = cv2.imread(train_path+emotion + "/" + frame) # open the image file
        try:
          dummy = img.shape
        except:
          i+=1
          print("Frame: " + frame + " is the number " + str(i) + " corrupted frame")
          print("removing: " + train_path+emotion+ "/" + frame)
          os.remove(train_path+emotion+ "/" + frame)

