from PIL import Image
import os.path

dataset = "./cleaned/"
train_path = dataset + "train_frames/"
val_path = dataset + "val_frames/"
test_path = dataset + "test_frames/"

emotions = os.listdir(test_path)

for emotion in emotions:
  frames = os.listdir(train_path+emotion)
  for frame in frames:
    img = os.path.join(train_path + emotion + "/" + frame)
    image = Image.open(img)
    if (image.size) != (112, 112):
      print(image.size)
