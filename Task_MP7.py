import subprocess
import os

os.chdir("./build")

detector_list = ["SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"]

for desc in detector_list:

    # subprocess.call("./2D_feature_tracking HARRIS")
    subprocess.getstatusoutput("./2D_feature_tracking " + desc)