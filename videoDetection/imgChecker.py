from PIL import Image
import os.path

dataset = "./cleaned/"
train_path = dataset + "train_frames/"
val_path = dataset + "val_frames/"
test_path = dataset + "test_frames/"

emotions = os.listdir(train_path)

for emotion in emotions:
  for frame in emotion:
    img = os.path.join(train_path + emotion + frame)
    image = Image.open( )
    print(image.size)
