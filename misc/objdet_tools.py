##################
## IMPORTS

## general package imports
import cv2
import numpy as np
import math
from shapely.geometry import Polygon

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

## Waymo open dataset reader
from simple_waymo_open_dataset_reader import utils as waymo_utils
from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2



##################

def get_rotation_matrix(roll, pitch, yaw):
    """ Convert Euler angles to a rotation matrix"""

    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)

    ones = np.ones_like(yaw)
    zeros = np.zeros_like(yaw)

    r_roll = np.stack([
        [ones,  zeros,     zeros],
        [zeros, cos_roll, -sin_roll],
        [zeros, sin_roll,  cos_roll]])

    r_pitch = np.stack([
        [ cos_pitch, zeros, sin_pitch],
        [ zeros,     ones,  zeros],
        [-sin_pitch, zeros, cos_pitch]])

    r_yaw = np.stack([
        [cos_yaw, -sin_yaw, zeros],
        [sin_yaw,  cos_yaw, zeros],
        [zeros,    zeros,   ones]])

    pose = np.einsum('ijhw,jkhw,klhw->ilhw',r_yaw,r_pitch,r_roll)
    pose = pose.transpose(2,3,0,1)
    return pose


def display_laser_on_image(img, pcl, vehicle_to_image):
    # Convert the pointcloud to homogeneous coordinates.
    pcl1 = np.concatenate((pcl,np.ones_like(pcl[:,0:1])),axis=1)

    # Transform the point cloud to image space.
    proj_pcl = np.einsum('ij,bj->bi', vehicle_to_image, pcl1) 

    # Filter LIDAR points which are behind the camera.
    mask = proj_pcl[:,2] > 0
    proj_pcl = proj_pcl[mask]
    proj_pcl_attr = pcl_attr[mask]

    # Project the point cloud onto the image.
    proj_pcl = proj_pcl[:,:2]/proj_pcl[:,2:3]

    # Filter points which are outside the image.
    mask = np.logical_and(
        np.logical_and(proj_pcl[:,0] > 0, proj_pcl[:,0] < img.shape[1]),
        np.logical_and(proj_pcl[:,1] > 0, proj_pcl[:,1] < img.shape[1]))

    proj_pcl = proj_pcl[mask]
    proj_pcl_attr = proj_pcl_attr[mask]

    # Colour code the points based on distance.
    coloured_intensity = 255*cmap(proj_pcl_attr[:,0]/30)

    # Draw a circle for each point.
    for i in range(proj_pcl.shape[0]):
        cv2.circle(img, (int(proj_pcl[i,0]),int(proj_pcl[i,1])), 1, coloured_intensity[i])

# LABELS AND OBJECTS

# extract object labels from frame
def validate_object_labels(object_labels, pcl, configs, min_num_points):

    ## Create initial list of flags where every object is set to `valid`
    valid_flags = np.ones(len(object_labels)).astype(bool)

    ## Mark labels as invalid that do not enclose a sufficient number of lidar points
    vehicle_to_labels = [np.linalg.inv(waymo_utils.get_box_transformation_matrix(label.box)) for label in object_labels] # for each label, compute transformation matrix from vehicle space to box space
    vehicle_to_labels = np.stack(vehicle_to_labels)

    pcl_no_int = pcl[:, :3] # strip away intensity information from point cloud 
    pcl1 = np.concatenate((pcl_no_int, np.ones_like(pcl_no_int[:, 0:1])), axis=1) # convert pointcloud to homogeneous coordinates    
    proj_pcl = np.einsum('lij,bj->lbi', vehicle_to_labels, pcl1) # transform point cloud to label space for each label (proj_pcl shape is [label, LIDAR point, coordinates])
    mask = np.logical_and.reduce(np.logical_and(proj_pcl >= -1, proj_pcl <= 1), axis=2) # for each pair of LIDAR point & label, check if point is inside the label's box (mask shape is [label, LIDAR point])

    counts = mask.sum(1) # count points inside each label's box and keep boxes which contain min. no of points
    valid_flags = counts >= min_num_points

    ## Mark labels as invalid which are ...
    for index, label in enumerate(object_labels):

        ## ... outside the object detection range
        label_obj = [label.type, label.box.center_x, label.box.center_y, label.box.center_z,
                     label.box.height, label.box.width, label.box.length, label.box.heading]
        valid_flags[index] = valid_flags[index] and is_label_inside_detection_area(label_obj, configs)                     

        ## ... flagged as "difficult to detect" or not of type "vehicle" 
        if(label.detection_difficulty_level > 0 or label.type != label_pb2.Label.Type.TYPE_VEHICLE):
            valid_flags[index] = False
        
    
    return valid_flags


# convert ground truth labels into 3D objects
def convert_labels_into_objects(object_labels, configs):
    
    detections = []
    for label in object_labels:
        # transform label into a candidate object
        if label.type==1 : # only use vehicles
            candidate = [label.type, label.box.center_x, label.box.center_y, label.box.center_z,
                         label.box.height, label.box.width, label.box.length, label.box.heading]

            # only add to object list if candidate is within detection area    
            if(is_label_inside_detection_area(candidate, configs)):
                detections.append(candidate)

    return detections


# compute location of each corner of a box and returns [front_left, rear_left, rear_right, front_right]
def compute_box_corners(x,y,w,l,yaw):
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    fl = (x - w / 2 * cos_yaw - l / 2 * sin_yaw,  # front left
          y - w / 2 * sin_yaw + l / 2 * cos_yaw)

    rl = (x - w / 2 * cos_yaw + l / 2 * sin_yaw,  # rear left
          y - w / 2 * sin_yaw - l / 2 * cos_yaw)

    rr = (x + w / 2 * cos_yaw + l / 2 * sin_yaw,  # rear right
          y + w / 2 * sin_yaw - l / 2 * cos_yaw)

    fr = (x + w / 2 * cos_yaw - l / 2 * sin_yaw,  # front right
          y + w / 2 * sin_yaw + l / 2 * cos_yaw)

    return [fl,rl,rr,fr]


# checks whether label is inside detection area
def is_label_inside_detection_area(label, configs, min_overlap=0.5):

    # convert current label object into Polygon object
    _, x, y, _, _, w, l, yaw = label
    label_obj_corners = compute_box_corners(x,y,w,l,yaw)
    label_obj_poly = Polygon(label_obj_corners)   

    # convert detection are into polygon
    da_w = (configs.lim_x[1] - configs.lim_x[0])  # width
    da_l = (configs.lim_y[1] - configs.lim_y[0])  # length
    da_x = configs.lim_x[0] + da_w/2  # center in x
    da_y = configs.lim_y[0] + da_l/2  # center in y
    da_corners = compute_box_corners(da_x,da_y,da_w,da_l,0)
    da_poly = Polygon(da_corners)   

    # check if detection area contains label object
    intersection = da_poly.intersection(label_obj_poly)
    overlap = intersection.area / label_obj_poly.area

    return False if(overlap <= min_overlap) else True



##################
# VISUALIZATION

# extract RGB front camera image and camera calibration
def extract_front_camera_image(frame):
    # extract camera and calibration from frame
    camera_name = dataset_pb2.CameraName.FRONT
    camera = waymo_utils.get(frame.images, camera_name)

    # get image and convert to RGB
    image = waymo_utils.decode_image(camera)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

# visualize object labels in camera image
def project_labels_into_camera(camera_calibration, image, labels, labels_valid, img_resize_factor=1.0):

    # get transformation matrix from vehicle frame to image
    vehicle_to_image = waymo_utils.get_image_transform(camera_calibration)

    # draw all valid labels
    for label, vis in zip(labels, labels_valid):
        if vis:
            colour = (0, 255, 0)
        else:
            colour = (255, 0, 0)

        # only show labels of type "vehicle"
        if(label.type == label_pb2.Label.Type.TYPE_VEHICLE):
            waymo_utils.draw_3d_box(image, vehicle_to_image, label, colour=colour)

    # resize image
    if (img_resize_factor < 1.0):
        width = int(image.shape[1] * img_resize_factor)
        height = int(image.shape[0] * img_resize_factor)
        dim = (width, height)
        img_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return img_resized
    else:
        return image


# visualize ground-truth labels as overlay in birds-eye view
def show_objects_labels_in_bev(detections, object_labels, bev_maps, configs):

    # project detections and labels into birds-eye view
    bev_map = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    bev_map = cv2.resize(bev_map, (configs.bev_width, configs.bev_height))
    
    label_detections = convert_labels_into_objects(object_labels, configs)
    project_detections_into_bev(bev_map, label_detections, configs, [0,255,0])
    project_detections_into_bev(bev_map, detections, configs, [0,0,255])
    

    bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
    cv2.imshow('labels (green) vs. detected objects (red)', bev_map)


def extract_bounding_boxes(frame):
    objects = []

    for camera_labels in frame.camera_labels:
        for label in camera_labels.labels:
            if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
                obj = {
                    'box': label.box,
                    'id': label.id,
                    'detection_difficulty_level': label.detection_difficulty_level,
                    'box_type': label.type
                }
                objects.append(obj)

    return objects
def compute_performance_stats(detections, labels):
    ious = []
    center_devs = []
    all_positives = len(labels)
    true_positives = 0
    false_negatives = 0
    false_positives = 0

    for label in labels:
        match_found = False
        for detection in detections:
            # Basit karşılaştırma: ID’ler eşleşiyorsa doğru eşleşme say
            if label['id'] == detection['id']:
                true_positives += 1
                match_found = True
                break
        if not match_found:
            false_negatives += 1

    false_positives = len(detections) - true_positives

    # sahte IOU ve center dev listeleri (şimdilik boş)
    return ious, center_devs, (all_positives, true_positives, false_negatives, false_positives)




