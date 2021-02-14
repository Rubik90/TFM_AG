import os
import numpy as np
import shutil

INPUT_DIR = "../../datasets/swapped"
OUTPUT_DIR = "../../datasets/random_sampled"

if (os.path.exists(f"{OUTPUT_DIR}")):
    shutil.rmtree(f"{OUTPUT_DIR}")

os.mkdir(f"{OUTPUT_DIR}")

unbalanced_splits = ["train_frames", "val_frames"]
test_split = "test_frames"
emotions = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]

for split in unbalanced_splits:
    os.mkdir(f"{OUTPUT_DIR}/{split}")

    for emotion in emotions:
        os.mkdir(f"{OUTPUT_DIR}/{split}/{emotions.index(emotion)}")
        frames = [f for f in os.listdir(f"{INPUT_DIR}/{split}/{emotion}/") if os.path.splitext(f)[1] == ".bmp"]
        random_sampled_frames = np.random.choice(frames, 394, replace = False)

        for frame in random_sampled_frames:
            shutil.copy(f"{INPUT_DIR}/{split}/{emotion}/{frame}", f"{OUTPUT_DIR}/{split}/{emotions.index(emotion)}")

# Copy the test split folder over the random sampled dataset directory
os.mkdir(f"{OUTPUT_DIR}/{test_split}")

for emotion in emotions:
    os.mkdir(f"{OUTPUT_DIR}/{test_split}/{emotions.index(emotion)}")
    frames = [f for f in os.listdir(f"{INPUT_DIR}/{test_split}/{emotion}/") if os.path.splitext(f)[1] == ".bmp"]

    for frame in frames:
        shutil.copy(f"{INPUT_DIR}/{test_split}/{emotion}/{frame}", f"{OUTPUT_DIR}/{test_split}/{emotions.index(emotion)}")
