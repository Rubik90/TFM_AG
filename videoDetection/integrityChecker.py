from os import listdir
from PIL import Image
import os

dataset = "./cleaned/"
train_path = dataset + "train_frames/"
val_path = dataset + "val_frames/"
test_path = dataset + "test_frames/"

emotions = os.listdir(train_path)

for emotion in emotions:
    frames = os.listdir(test_path+emotion)
    for frame in frames:
      if frame.endswith('.bmp'):
        try:
          img = Image.open('./'+frame) # open the image file
          img.verify() # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
          print('Bad file:', frame) # print out the names of corrupt files
          os.remove(test_path+emotion+"/"+frame)
