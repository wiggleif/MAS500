# This script obtains all filenames inside the path: data/augall/  and writes a new file with all those filnames. The generated file is used to localize the path of every image during a training-session in darknet

import os

image_files = []
os.chdir(os.path.join("data", "augall"))
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".png"):
        image_files.append("data/augall/" + filename)
os.chdir("..")
with open("train.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")
