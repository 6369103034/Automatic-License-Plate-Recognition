import cv2
import pytesseract
import imutils
import numpy as np
import os

# --- Adjust this path to where Tesseract is installed on your system ---
# For example: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = imutils.resize(img, width=600)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)      # reduce noise
    edged = cv2.Canny(gray, 30, 200)                  # edge detection
    return img, gray, edged

def find_plate_contour(edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            return approx
    return None

def extract_plate(img, gray, contour):
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    x, y, w, h = cv2.boundingRect(contour)
    plate_img = new_image[y:y + h, x:x + w]
    return plate_img, (x, y, w, h)

def ocr_plate(plate_img):
    if plate_img is None or plate_img.size == 0:
        return ""
    # Enlarge for better OCR
    plate_img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    # Improve contrast and binarize
    gray_plate = cv2.equalizeHist(gray_plate)
    _, thresh = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # OCR configuration: single line, capital letters & digits only
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(thresh, config=config)
    return ''.join(filter(str.isalnum, text))

def main(image_path):
    img, gray, edged = preprocess_image(image_path)
    contour = find_plate_contour(edged)

    if contour is None:
        print("No plate contour detected.")
        return

    plate_img, bbox = extract_plate(img, gray, contour)
    plate_text = ocr_plate(plate_img)
    x, y, w, h = bbox

    # Draw bounding box and text
    label = plate_text if plate_text else "No Text"
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    print("Detected Plate Number:", plate_text if plate_text else "(none)")
    cv2.imshow("Plate Region", plate_img)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change this to your actual image file name or full path
    image_path = "car.jpg"
    if not os.path.exists(image_path):
        print(f"Image '{image_path}' not found in current directory.")
    else:
        main(image_path)
