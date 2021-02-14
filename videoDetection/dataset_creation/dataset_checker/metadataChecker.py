from os import listdir
from PIL import Image
import os
import cv2

folder = "./swapped/"

train_path = folder + "train_frames/"
val_path = folder + "val_frames/"
test_path = folder + "test_frames/"
sets = [train_path, val_path, test_path]

i=0
for dataset in sets:
  emotions = os.listdir(dataset)
  for emotion in emotions:
    frames = os.listdir(dataset + emotion)
    for frame in frames:
      if frame.endswith('.bmp'):
        img = cv2.imread(dataset+emotion + "/" + frame) # open the image file
        try:
          dummy = img.shape
        except:
          i+=1
          print("Frame: " + frame + " is the number " + str(i) + " corrupted frame")
          print("removing: " + train_path+emotion+ "/" + frame)
          os.remove(train_path+emotion+ "/" + frame)
