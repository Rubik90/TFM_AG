import subprocess
import os
import shutil
def split(audio,output):
    command = f"ffmpeg -i '{audio}' -f segment -segment_time 0.0416666666666667 -c copy {output}"
    subprocess.call(command,shell=True)

split('audio.wav','out%03d.wav')
elements = os.listdir('./')

if(os.path.isdir("./audioChunks/")):
  shutil.rmtree("./audioChunks/")

os.mkdir("./audioChunks/")

for element in elements:
  if element[:3]=="out":
    shutil.copyfile('./'+element, './audioChunks/'+element)
    os.remove(element)