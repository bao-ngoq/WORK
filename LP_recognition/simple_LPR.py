import cv2
import numpy as np
import pytesseract

def preprocess_image(image_path):
    # Đọc ảnh
    image = cv2.imread(image_path)
    
    # Chuyển đổi sang thang độ xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Làm mờ ảnh để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Phát hiện biên bằng Canny
    edged = cv2.Canny(blurred, 50, 150)
    
    return image, edged

def find_license_plate(image, edged):
    # Tìm các contour trong ảnh
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Lặp qua các contour để tìm biển số xe
    for contour in contours:
        # Tính toán xấp xỉ contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Nếu contour có 4 đỉnh, có thể là biển số xe
        if len(approx) == 4:
            license_plate = approx
            break

    # Vẽ contour lên ảnh gốc
    cv2.drawContours(image, [license_plate], -1, (0, 255, 0), 3)
    
    return image, license_plate

def recognize_text(image, license_plate):
    # Tạo mask để cắt biển số xe
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [license_plate], -1, 255, -1)
    
    # Áp dụng mask lên ảnh gốc
    masked = cv2.bitwise_and(image, image, mask=mask)
    
    # Cắt vùng biển số xe
    (x, y, w, h) = cv2.boundingRect(license_plate)
    cropped = masked[y:y+h, x:x+w]
    
    # Chuyển đổi sang thang độ xám
    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    # Nhận diện văn bản bằng Tesseract OCR
    text = pytesseract.image_to_string(gray_cropped, config='--psm 8')
    
    return text

def main(image_path):
    # Tiền xử lý ảnh
    image, edged = preprocess_image(image_path)
    
    # Tìm biển số xe
    image, license_plate = find_license_plate(image, edged)
    
    # Nhận diện văn bản
    text = recognize_text(image, license_plate)
    
    # Hiển thị kết quả
    print("Biển số xe nhận diện được:", text)
    cv2.imshow("License Plate", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("/mnt/d/SPARK/LP_IMAGE/test_LP_2.png")
