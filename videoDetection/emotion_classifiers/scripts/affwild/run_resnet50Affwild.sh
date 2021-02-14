cd ../classifiers
python3 classifier_launcher.py --train_path "../../datasets/random_sampled/test_frames" --val_path "../../datasets/random_sampled/val_frames" --test_path "../../datasets/random_sampled/train_frames" --model "resnet50"
