import cv2
import numpy as np

def cartoon_ps2_style(image):
    # Step 1: Reduce the resolution (ลดความละเอียดเพื่อเพิ่มความเร็ว)
    scale = 0.5  # ลดขนาดภาพลงครึ่งหนึ่ง
    small_image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    image = cv2.resize(small_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Step 2: Apply bilateral filter for smooth colors (ลดค่าพารามิเตอร์ของ Bilateral Filter)
    color = cv2.bilateralFilter(image, d=7, sigmaColor=100, sigmaSpace=100)

    # Step 3: Reduce color palette (Color Quantization)
    Z = color.reshape((-1, 3))
    Z = np.float32(Z)

    # Reduce number of colors with k-means clustering
    K = 10  # ลดจำนวนสีให้เหลือ 4 สี
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized_image = centers[labels.flatten()]
    quantized_image = quantized_image.reshape((image.shape))

    # Step 4: Detect edges for "toon-style" outlines
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Step 5: Combine quantized colors and edges
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert edges to 3 channels
    cartoon = cv2.subtract(quantized_image, edges)  # Combine quantized image with edges

    return cartoon

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize input frame for better performance (ลดขนาดเฟรมก่อนประมวลผล)
    frame = cv2.resize(frame, (640, 480))  # ลดขนาดภาพให้เหลือ 640x480

    # Apply cartoon PS2-style effect
    cartoon_frame = cartoon_ps2_style(frame)

    # Display the original and cartoon-style frames
    cv2.imshow('Original Video', frame)
    cv2.imshow('Cartoon PS2 Style', cartoon_frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
