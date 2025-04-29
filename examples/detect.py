import os
import cv2
import io
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2

# Ayarlar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval().to(device)

# Dataset yolu
DATA_PATH = "dataset/training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord"
OUTPUT_DIR = "detections_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Veri dönüştürme
transform = transforms.Compose([
    transforms.ToTensor()
])

# TFRecord dosyasını oku
reader = WaymoDataFileReader(DATA_PATH)
data_iter = iter(reader)

# IoU hesaplama
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# Sadece birkaç frame işleyelim
MAX_FRAMES = 50

all_precisions = []
all_recalls = []

for i in range(MAX_FRAMES):
    frame = next(data_iter)

    # FRONT görüntüyü al
    camera_image = [img for img in frame.images if img.name == dataset_pb2.CameraName.FRONT][0]
    img = Image.open(io.BytesIO(camera_image.image))
    img_rgb = np.array(img)
    tensor_img = transform(img).to(device)

    # Predict
    with torch.no_grad():
        prediction = model([tensor_img])[0]

    # Prediction filtreleme (sadece skor > 0.5 ve 'car', 'truck')
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()

    selected_boxes = []
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5 and label in [3, 8]:  # 3 = car, 8 = truck
            selected_boxes.append(box)

    # Ground truth kutuları
    gt_boxes = []
    for camera_label in frame.camera_labels:
        if camera_label.name == dataset_pb2.CameraName.FRONT:
            for label in camera_label.labels:
                box = label.box
                cx, cy, w, h = box.center_x, box.center_y, box.length, box.width
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                x2 = int(cx + w / 2)
                y2 = int(cy + h / 2)
                gt_boxes.append([x1, y1, x2, y2])

    # IoU bazlı eşleştirme
    TP, FP, FN = 0, 0, 0
    matched = set()

    for pred_box in selected_boxes:
        match_found = False
        for j, gt in enumerate(gt_boxes):
            if j in matched:
                continue
            iou = compute_iou(pred_box, gt)
            if iou > 0.5:
                TP += 1
                matched.add(j)
                match_found = True
                break
        if not match_found:
            FP += 1

    FN = len(gt_boxes) - TP
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    all_precisions.append(precision)
    all_recalls.append(recall)

    # Görsel çizim ve kayıt
    img_draw = img_rgb.copy()
    for box in gt_boxes:
        cv2.rectangle(img_draw, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    for box in selected_boxes:
        cv2.rectangle(img_draw, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

    save_path = os.path.join(OUTPUT_DIR, f"frame_{i:03d}.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))
    print(f"[{i}] Saved to {save_path} | Precision: {precision:.2f}, Recall: {recall:.2f}")

# PR Curve
plt.plot(all_recalls, all_precisions, marker='o')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.savefig(os.path.join(OUTPUT_DIR, "pr_curve.png"))
print("PR curve saved.")

# Ortalama AP (approx mAP@50)
mean_ap = np.mean(all_precisions)
print(f"\n✅ Approximate mAP@50: {mean_ap:.4f}")
