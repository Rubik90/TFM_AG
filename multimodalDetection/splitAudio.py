import subprocess
import os
import shutil
def extract_audio(video,output):
    command = f"ffmpeg -i '{video}' -f segment -segment_time 1 -c copy {output}"
    subprocess.call(command,shell=True)

extract_audio('test.wav','out%03d.wav')
elements = os.listdir('./')

if(os.path.isdir("./multimodalDetection/audioChunks/")):
  shutil.rmtree("./multimodalDetection/audioChunks/")

os.mkdir("./multimodalDetection/audioChunks/")

for element in elements:
  if element[:3]=="out":
    shutil.copyfile('./'+element, './multimodalDetection/audioChunks/'+element)
    os.remove(element)