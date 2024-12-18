import cv2
import numpy as np

def ps2_graphics_effect(image):
    # Step 1: Reduce resolution (Low Resolution Simulation)
    scale = 0.5  # ลดขนาดภาพลง 50%
    low_res = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    low_res = cv2.resize(low_res, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Step 2: Apply Color Quantization (Reduce Color Depth)
    Z = low_res.reshape((-1, 3))
    Z = np.float32(Z)

    K = 4  # จำนวนสี (ปรับได้)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized_image = centers[labels.flatten()]
    quantized_image = quantized_image.reshape((low_res.shape))

    # Step 3: Enhance edges (Optional, for a sharper look)
    gray = cv2.cvtColor(quantized_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)  # ตรวจจับเส้นขอบ
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # แปลงเป็น 3 ช่องสี

    # Step 4: Combine quantized image with edges
    ps2_effect = cv2.addWeighted(quantized_image, 0.9, edges, 0.1, 0)

    return ps2_effect

# โหลดภาพ
image_path = 'gus.jpg'  # ใส่พาธของภาพที่ต้องการ
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
    exit()

# Apply PS2 graphics effect
ps2_image = ps2_graphics_effect(image)

# แสดงผลภาพต้นฉบับและภาพที่มีกราฟิกสไตล์ PS2
cv2.imshow('Original Image', image)
cv2.imshow('PS2 Graphics Effect', ps2_image)

# รอจนกว่าจะกด 'q' เพื่อปิดหน้าต่าง
cv2.waitKey(0)
cv2.destroyAllWindows()
