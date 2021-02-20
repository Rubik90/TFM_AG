import os

command = f"ffmpeg -i predictions.avi -i audio.wav -c copy -map 0:v:0 -map 1:a:0 predictionsAudio.avi"
os.system(command)