import os
import numpy as np
import shutil
import math

INPUT_DIR = "../../datasets/cleaned"
OUTPUT_DIR = "../../datasets/balanced"

splits = ["train_frames", "val_frames", "test_frames"]
emotions = ["Anger", "Disgust", "Fear",
            "Happiness", "Neutral", "Sadness", "Surprise"]

ratios = [0.70, 0.15, 0.15]

if (os.path.exists(f"{OUTPUT_DIR}")):
    shutil.rmtree(f"{OUTPUT_DIR}")

os.mkdir(f"{OUTPUT_DIR}")

for emotion in emotions:
    os.mkdir(f"{OUTPUT_DIR}/{emotion}")

for split in splits:
    for emotion in emotions:

        frames = [f for f in os.listdir(
            f"{INPUT_DIR}/{split}/{emotion}/") if os.path.splitext(f)[1] == ".bmp"]

        for frame in frames:
            shutil.copy(f"{INPUT_DIR}/{split}/{emotion}/{frame}",
                        f"{OUTPUT_DIR}/{emotion}")

for emotion in emotions:
    frames = [f for f in os.listdir(
        f"{OUTPUT_DIR}/{emotion}/") if os.path.splitext(f)[1] == ".bmp"]
    num_frames = len(frames)
    prev_bound = 0

    for index, split in enumerate(splits):
        bound = prev_bound + min(math.ceil(num_frames * ratios[index]), num_frames)
        split_frames = frames[prev_bound:bound]
        prev_bound = bound

        if (not os.path.exists(f"{OUTPUT_DIR}/{split}")):
            os.mkdir(f"{OUTPUT_DIR}/{split}")
        
        if (not os.path.exists(f"{OUTPUT_DIR}/{split}/{emotion}")):
            os.mkdir(f"{OUTPUT_DIR}/{split}/{emotion}")

        for frame in split_frames:
            shutil.copy(f"{OUTPUT_DIR}/{emotion}/{frame}", f"{OUTPUT_DIR}/{split}/{emotion}")
	
    shutil.rmtree(f"{OUTPUT_DIR}/{emotion}")