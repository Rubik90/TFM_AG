import subprocess, os

os.chdir('../')
file_ = open('./results/shell_conv2dFer.txt', 'w+') 
os.chdir('./classifiers')
subprocess.run('python3 conv2dfer.py', shell=True, stdout=file_) 
file_.close()
