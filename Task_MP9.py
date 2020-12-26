import subprocess
import os
import csv

with open('TASK MP.9.csv', mode='w+') as Task:
    writer = csv.writer(Task, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["Detector/Descriptor Combination","Average Time Taken To Detect & Extract Keypoints"])

os.chdir("./build")

detector_list = ["SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"]
descriptors_list = ["BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"]
for det in detector_list:
    for desc in descriptors_list:
    # subprocess.call("./2D_feature_tracking HARRIS")
        subprocess.getstatusoutput("./2D_feature_tracking " + det + " "+desc)