from PIL import Image
import pytesseract
import cv2
import numpy as np

# Mở hình ảnh
image = Image.open("/mnt/d/SPARK/LP_IMAGE/test_LP_OCR_2.jpg")
image = image.convert('L')  # Chuyển sang ảnh đen trắng

# Chuyển đổi ảnh sang mảng numpy
image_cv = np.array(image)

# Làm sắc nét ảnh (ví dụ sử dụng thresholding)
_, thresh_img = cv2.threshold(image_cv, 150, 255, cv2.THRESH_BINARY)

# Chuyển ảnh đã xử lý sang PIL Image để sử dụng với pytesseract
processed_image = Image.fromarray(thresh_img)

# Nhận diện văn bản từ hình ảnh
text = pytesseract.image_to_string(processed_image, lang='eng', config='--psm 11')

# In kết quả
print(text)
