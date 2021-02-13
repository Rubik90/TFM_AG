import subprocess

file_ = open('./results/shell_vgg16Fer.txt', 'w+') 
subprocess.run('python3 vgg16Fer.py', shell=True, stdout=file_) 
file_.close() 