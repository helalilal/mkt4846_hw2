# imports
import numpy as np
import matplotlib
# matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well     
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib import colors
from matplotlib.transforms import Affine2D
import matplotlib.ticker as ticker
import os
import cv2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from tools.waymo_reader.simple_waymo_open_dataset_reader import label_pb2
        
        
def make_movie(path):
    # read track plots
    images = [img for img in os.listdir(path) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(path, images[0]))
    height, width, layers = frame.shape

    # save with 10fps to result dir
    video = cv2.VideoWriter(os.path.join(path, 'my_results.avi'), 0, 10, (width,height))

    for image in images:
        fname = os.path.join(path, image)
        video.write(cv2.imread(fname))
        os.remove(fname) # clean up

    cv2.destroyAllWindows()
    video.release()