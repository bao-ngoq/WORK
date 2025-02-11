from pyspark import SparkContext
from pyspark.streaming import StreamingContext
import cv2
import numpy as np

# Khởi tạo SparkContext và StreamingContext
sc = SparkContext(appName="VideoStreamProcessing")
ssc = StreamingContext(sc, batchDuration=1)  # Batch duration: 1 giây

# Hàm xử lý từng khung hình
def process_frame(frame):
    # Chuyển đổi frame thành ảnh xám
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray_frame

# Hàm đọc video và chia thành các khung hình
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

# Tạo DStream từ video
video_path = "/mnt/d/SPARK/video_test.avi"
frames = read_video(video_path)
frames_rdd = sc.parallelize(frames)
frames_stream = ssc.queueStream([frames_rdd])

# Áp dụng hàm xử lý lên từng khung hình
processed_stream = frames_stream.map(process_frame)

# Hiển thị kết quả (ví dụ: lưu các khung hình đã xử lý)
def save_frame(rdd):
    frames = rdd.collect()
    for i, frame in enumerate(frames):
        cv2.imwrite(f"processed_frame_{i}.jpg", frame)

processed_stream.foreachRDD(save_frame)

# Bắt đầu Spark Streaming
ssc.start()
ssc.awaitTermination()
