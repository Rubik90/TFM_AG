import subprocess

file_ = open('./results/shell_resnetFer.txt', 'w+') 
subprocess.run('python3 resnet50Fer.py', shell=True, stdout=file_) 
file_.close() 