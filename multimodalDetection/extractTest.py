import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import moviepy.editor
import sys
import shutil


vid = moviepy.editor.VideoFileClip("./test.avi")
audio = vid.audio
audio.write_audiofile("./test" + ".wav")
shutil.copyfile("./test.wav","./multimodalDetection/test.wav")
shutil.copyfile("./test.avi","./multimodalDetection/test.avi")
