import subprocess, os
#comment the first two lines and uncomment the rest of the snippet code to save the entire output on file
os.chdir('../classifiers')
os.system('python3 resnet50Fer.py')

"""
os.chdir('../')
file_ = open('./results/shell_resnetFer.txt', 'w+')
os.chdir('./classifiers')
subprocess.run('python3 resnet50Fer.py', shell=True, stdout=file_) 
file_.close()
"""
