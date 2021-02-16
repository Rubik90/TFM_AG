cd ../../classifiers
python3 classifier_launcher.py --train_path "../../datasets/shuffled_balanced/train_frames" --val_path "../../datasets/shuffled_balanced/val_frames" --test_path "../../datasets/shuffled_balanced/test_frames" --model "vgg16"
