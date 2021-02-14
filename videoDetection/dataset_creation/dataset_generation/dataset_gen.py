import os
import sys
import shutil
import glob
from PIL import Image

if "--videos_path" in sys.argv:
    videos_path = sys.argv[sys.argv.index("--videos_path") + 1]
else:
    print("ERROR: No value specified for the frames path")
    sys.exit()

if "--annotations_path" in sys.argv:
    annotations_path = sys.argv[sys.argv.index("--annotations_path") + 1]
else:
    print("ERROR: No value specified for the annotations path")
    sys.exit()

if "--dataset_path" in sys.argv:
    dataset_path = sys.argv[sys.argv.index("--dataset_path") + 1]
else:
    print("ERROR: No value specified for the destination path where the dataset will be created")
    sys.exit()

videos = os.listdir(videos_path)
videos.sort()

annotations = os.listdir(annotations_path)
annotations.sort()

emotions = ['Neutral', 'Anger', 'Disgust',
            'Fear', 'Happiness', 'Sadness', 'Surprise']

if(os.path.isdir(dataset_path)):
    shutil.rmtree(dataset_path)

os.mkdir(dataset_path)

for emotion in emotions:
    path = f'{dataset_path}/{emotion}'

    if(os.path.isdir(path)):
        shutil.rmtree(path)

    os.mkdir(path)

print('Generating the dataset, please wait...\n')

for video in videos:
    for annotation in annotations:
        video_id = video[:-8]
        annotation_id = annotation[:-4]

        if (video_id == annotation_id):
            print(f'Currently processing video \'{video_id}\'')

            # Get the annotations
            with open(annotations_path + annotation, 'r') as ann:
                lines = ann.readlines()

            # Get the frames
            frames = os.listdir(videos_path + video)
            frames.sort()

            for i in range(1, len(frames) - 1):
                annotation_value = int(lines[i])
                
                print("", end=f"\rProgress: [{i + 1}/{len(frames) - 1}]")

                if (annotation_value >= 0 and annotation_value <= 6):  # Skip untagged frames
                    shutil.copy2(f'{videos_path}/{video}/{frames[i]}', f'{dataset_path}/{emotions[int(lines[i])]}')
            print("\n")
print("")
