import os
import sys
sys.path.append(os.getcwd())

from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2
from s2025 import midterm

# Burada tam dosya yolunu veriyoruz:
filename = '/home/hilal-2004/simple-waymo-open-dataset-reader/examples/dataset/training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'

# Waymo verisini oku
datafile = WaymoDataFileReader(filename)

# İlk frame'i al
frame = next(datafile)

# Kullanacağımız lidar sensörü
lidar_name = dataset_pb2.LaserName.TOP

# Test fonksiyonları
midterm.print_no_of_vehicles(frame)
midterm.print_vfov_lidar(frame, lidar_name)
midterm.get_max_min_range(frame, lidar_name)
midterm.print_pitch_resolution(frame, lidar_name)

# Görselleştirme fonksiyonları (ekran açılır)
midterm.vis_range_channel(frame, lidar_name)
midterm.vis_intensity_channel(frame, lidar_name)
