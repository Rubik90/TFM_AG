import subprocess
import os
import shutil

elements = os.listdir('./')

for element in elements:
  if element[:3]=="out":
    os.remove(element)