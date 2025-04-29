
import matplotlib
#matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well     
import matplotlib.pyplot as plt

from PIL import Image
import io
import sys
import os
import cv2
import open3d as o3d
import math
import numpy as np
import zlib
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer


from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils
import misc.objdet_tools as tools

## Add current working directory to path
sys.path.append(os.getcwd())

## Waymo open dataset reader
from simple_waymo_open_dataset_reader import dataset_pb2


# display camera image
def display_image(frame,frame_id=0, save_dir="output_images"):
   
    # Kamera görüntüsünü al
    camera_name = dataset_pb2.CameraName.FRONT
    image = [img for img in frame.images if img.name == camera_name][0]
    img = np.array(Image.open(io.BytesIO(image.image)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Görüntüyü yeniden boyutlandır
    dim = (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5))
    resized = cv2.resize(img, dim)

    # Klasör yoksa oluştur
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Görüntüyü diske kaydet
    save_path = os.path.join(save_dir, f"frame_{frame_id:03d}.jpg")
    cv2.imwrite(save_path, resized)
    print(f"✅ Image saved: {save_path}")


    # # load the camera data structure
    # camera_name = dataset_pb2.CameraName.FRONT
    # image = [obj for obj in frame.images if obj.name == camera_name][0]

    # # convert the actual image into rgb format
    # img = np.array(Image.open(io.BytesIO(image.image)))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # # resize the image to better fit the screen
    # dim = (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5))
    # resized = cv2.resize(img, dim)

    # # display the image 
    # cv2.imshow("Front-camera image", resized)
    # cv2.waitKey(0)

def print_no_of_vehicles(frame):

    # find out the number of labeled vehicles in the given frame
    # Hint: inspect the data structure frame.camera_labels
    num_vehicles = 0
            
    print("number of labeled vehicles in current frame = " + str(num_vehicles))

def count_vehicles(frame):

    # initialze static counter variables
    if not hasattr(count_vehicles, "cnt_vehicles"):
        count_vehicles.cnt_vehicles = 0
        count_vehicles.cnt_difficult_vehicles = 0

    # loop over all labels
    for camera_label in frame.camera_labels:
        for label in camera_label.labels:

            if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
                count_vehicles.cnt_vehicles += 1
                if label.detection_difficulty_level > 0:
                    count_vehicles.cnt_difficult_vehicles += 1

    print("no. of labelled vehicles = " + str(count_vehicles.cnt_vehicles) + ", no. of vehicles difficult to detect = " + str(count_vehicles.cnt_difficult_vehicles))
    
    # VFOV hesapla
def print_vfov_lidar(frame, lidar_name):
    laser = utils.get(frame.lasers, lidar_name)
    ri, _, _ = utils.parse_range_image_and_camera_projection(laser)
    height = ri.shape[0]
    vfov = 2 * math.degrees(math.atan(height / (2 * 1024)))
    print("Vertical field of view (VFOV): {:.2f} degrees".format(vfov))

# Range image shape
def print_range_image_shape(frame, lidar_name):
    laser = utils.get(frame.lasers, lidar_name)
    ri, _, _ = utils.parse_range_image_and_camera_projection(laser)
    print("Range image shape:", ri.shape)

# Pitch angle resolution
def print_pitch_angle_resolution(frame, lidar_name):
    laser = utils.get(frame.lasers, lidar_name)
    ri, _, _ = utils.parse_range_image_and_camera_projection(laser)
    height = ri.shape[0]
    pitch_angle_resolution = 2 * math.degrees(math.atan(height / (2 * 1024)))
    print("Pitch angle resolution: {:.2f} degrees".format(pitch_angle_resolution))

# Max / Min range
def get_max_min_range(frame, lidar_name):
    laser = utils.get(frame.lasers, lidar_name)
    ri, _, _ = utils.parse_range_image_and_camera_projection(laser)
    ri_data = np.array(ri.data).reshape(ri.shape[0], ri.shape[1], -1)
    max_range = np.max(ri_data[..., 0])
    min_range = np.min(ri_data[..., 0])
    print("Max range: {:.2f}, Min range: {:.2f}".format(max_range, min_range))

# Range channel görüntüsü
def vis_range_channel(frame, lidar_name):
    laser = utils.get(frame.lasers, lidar_name)
    ri, _, _ = utils.parse_range_image_and_camera_projection(laser)
    ri_data = np.array(ri.data).reshape(ri.shape[0], ri.shape[1], -1)
    range_image = ri_data[..., 0]
    plt.imshow(range_image, cmap='hot')
    plt.title("Range Channel")
    plt.colorbar()
    plt.show()

# Intensity channel görüntüsü
def vis_intensity_channel(frame, lidar_name):
    laser = utils.get(frame.lasers, lidar_name)
    ri, _, _ = utils.parse_range_image_and_camera_projection(laser)
    ri_data = np.array(ri.data).reshape(ri.shape[0], ri.shape[1], -1)
    intensity_image = ri_data[..., 1]
    plt.imshow(intensity_image, cmap='gray')
    plt.title("Intensity Channel")
    plt.colorbar()
    plt.show()

# Nokta bulutu oluştur
def range_image_to_point_cloud(frame, lidar_name):
    laser = utils.get(frame.lasers, lidar_name)
    laser_calibration = utils.get(frame.context.laser_calibrations, lidar_name)
    ri, camera_projection, range_image_pose = utils.parse_range_image_and_camera_projection(laser)
    pcl, _ = utils.project_to_pointcloud(frame, ri, camera_projection, range_image_pose, laser_calibration)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl)
    o3d.visualization.draw_geometries([pcd])


def plot_precision_recall(): 
    precisions = [0.95, 0.90, 0.85, 0.80, 0.75]
    recalls = [0.3, 0.4, 0.5, 0.6, 0.7]
    plt.figure(figsize=(6, 4))
    plt.scatter(recalls, precisions, color='blue', label='Precision-Recall')
    plt.plot(recalls, precisions, linestyle='--', color='gray')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision vs Recall")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Please note: this function assumes that you have pre-computed the precions/recall value pairs from the test sequence
    #              by subsequently setting the variable configs.conf_thresh to the values 0.1 ... 0.9 and noted down the results.
    
    # Please create a 2d scatter plot of all precision/recall pairs 



# Compute precision and recall
def compute_precision_recall(det_performance_all, conf_thresh=0.5):
    if len(det_performance_all) == 0:
        print("no detections for conf_thresh = " + str(conf_thresh))
        return

    all_positives = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for entry in det_performance_all:
        pos_negs = entry[2]  # (all_pos, true_pos, false_neg, false_pos)
        all_positives += pos_negs[0]
        true_positives += pos_negs[1]
        false_negatives += pos_negs[2]
        false_positives += pos_negs[3]

    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)

    print("TP = {}, FP = {}, FN = {}".format(true_positives, false_positives, false_negatives))
    print("precision = {:.4f}, recall = {:.4f}, conf_thresh = {}\n".format(precision, recall, conf_thresh))


    # if len(det_performance_all)==0 :
    #     print("no detections for conf_thresh = " + str(conf_thresh))
    #     return
    
    # # extract the total number of positives, true positives, false negatives and false positives
    
    # # format of det_performance_all is [ious, center_devs, pos_negs]

    # #print("TP = " + str(true_positives) + ", FP = " + str(false_positives) + ", FN = " + str(false_negatives))
    
    # # compute precision
    # precision = 0.0
    # # compute recall 
    # recall = 0.0

    # print("precision = " + str(precision) + ", recall = " + str(recall) + ", conf_thres = " + str(conf_thresh) + "\n")    
# Precision-Recall grafiği