import subprocess

file_ = open('./results/shell_conv2dFer.txt', 'w+') 
subprocess.run('python3 conv2dFer.py', shell=True, stdout=file_) 
file_.close() 