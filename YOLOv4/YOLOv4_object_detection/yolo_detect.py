import cv2
import numpy as np

# Load model YOLOv4
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load danh sách tên lớp từ coco.names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Đọc ảnh cần nhận diện
image = cv2.imread("test.jpg")
height, width, _ = image.shape

# Tiền xử lý ảnh
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Chạy YOLO để dự đoán
outs = net.forward(output_layers)

# Xử lý kết quả
class_ids, confidences, boxes = [], [], []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
            x, y = int(center_x - w / 2), int(center_y - h / 2)
            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Áp dụng Non-Maximum Suppression (NMS) để loại bỏ các khung trùng lặp
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Hiển thị kết quả
for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    color = (0, 255, 0)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Hiển thị ảnh
cv2.imshow("YOLOv4 Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
