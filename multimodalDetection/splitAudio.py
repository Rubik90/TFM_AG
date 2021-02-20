import subprocess
import os
import shutil
def extract_audio(video,output):
    command = f"ffmpeg -i '{video}' -f segment -segment_time 0.041666666 -c copy {output}"
    subprocess.call(command,shell=True)

extract_audio('audio.wav','out%03d.wav')
elements = os.listdir('./')

if(os.path.isdir("./audioChunks/")):
  shutil.rmtree("./audioChunks/")

os.mkdir("./audioChunks/")

for element in elements:
  if element[:3]=="out":
    shutil.copyfile('./'+element, './audioChunks/'+element)
    os.remove(element)