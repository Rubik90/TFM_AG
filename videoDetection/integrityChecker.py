from os import listdir
from PIL import Image
import os

dataset = "./cleaned/"

train_path = dataset + "train_frames/"
val_path = dataset + "val_frames/"
test_path = dataset + "test_frames/"
sets = [train_path, val_path, test_path]

for dataset in sets:
  emotions = os.listdir(dataset)
  for emotion in emotions:
    frames = os.listdir(train_path+emotion)
    i=0
    for frame in frames:
      if frame.endswith('.bmp'):
        try:
          img = Image.open(train_path+emotion + "/" + frame) # open the image file
          img.verify() # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
          print('Bad file:', frame, e) # print out the names of corrupt files
          i+=1
          print("Frame: " + frame + " is the number " + i +" corrupted frame")
