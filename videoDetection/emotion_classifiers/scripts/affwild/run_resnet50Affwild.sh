cd ../../classifiers
python3 classifier_launcher.py --train_path "/content/shuffled_balanced/train_frames" --val_path "/content/shuffled_balanced/val_frames" --test_path "/content/shuffled_balanced/test_frames" --model "resnet50"
