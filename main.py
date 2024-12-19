import cv2
import numpy as np

facemark = cv2.face.createFacemarkLBF()
model_path = "Data/lbfmodel.yaml"
facemark.loadModel(model_path)

# Cartoon PS2-style effect function
def cartoon_ps2_style(image):
    scale = 0.5  # Downscale for processing
    small_image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    image = cv2.resize(small_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Bilateral filter for smooth colors
    color = cv2.bilateralFilter(image, d=5, sigmaColor=75, sigmaSpace=75)

    # Color quantization (reduce color palette)
    Z = color.reshape((-1, 3))
    Z = np.float32(Z)
    K = 8  # Fewer colors for faster quantization
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized_image = centers[labels.flatten()]
    quantized_image = quantized_image.reshape((image.shape))

    # Detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Combine quantized colors and edges
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.subtract(quantized_image, edges)

    return cartoon

def detect_expression(landmarks):
    if len(landmarks) < 68:
        return "NEUTRAL"

    mouth = landmarks[48:68]
    mouth_width = np.linalg.norm(mouth[0] - mouth[6])  # Horizontal distance
    mouth_height = np.linalg.norm(mouth[3] - mouth[9])  # Vertical distance

    mouth_ratio = mouth_height / mouth_width
    # print(f"mouth_width: {mouth_width}, mouth_height: {mouth_height}, mouth_ratio: {mouth_ratio}")  # Debug print

    if mouth_ratio > 0.6:
        return "SURPRISED"
    elif mouth_ratio > 0.45:
        return "SMILING"
    else:
        return "NEUTRAL"
    
def add_comic_bubble(frame, face, expression, bubble_image_path):
    x, y, w, h = face
    bubble_image = cv2.imread(bubble_image_path, cv2.IMREAD_UNCHANGED)
    if bubble_image is None:
        print("Error: Comic bubble image not found.")
        return

    bubble_width = int(w * 1.5)
    bubble_height = int(h * 0.75)
    resized_bubble = cv2.resize(bubble_image, (bubble_width, bubble_height), interpolation=cv2.INTER_AREA)

    bubble_x = x + w // 2 - bubble_width // 2
    bubble_y = y - bubble_height - 10
    bubble_x = max(0, min(bubble_x, frame.shape[1] - bubble_width))
    bubble_y = max(0, min(bubble_y, frame.shape[0] - bubble_height))

    for c in range(3):
        alpha = resized_bubble[:, :, 3] / 255.0
        frame[bubble_y:bubble_y + bubble_height, bubble_x:bubble_x + bubble_width, c] = \
            (alpha * resized_bubble[:, :, c] +
             (1 - alpha) * frame[bubble_y:bubble_y + bubble_height, bubble_x:bubble_x + bubble_width, c])

    text = {"SMILING": "SKIBIDI RIZZ !",
            "SURPRISED": "IT'S SO BIG...",
            "NEUTRAL": "SIGMA MODE"}.get(expression, "...")
    
    # Calculate the text size and position to center it
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = bubble_x + (bubble_width - text_size[0]) // 2
    text_y = bubble_y + (bubble_height + text_size[1]) // 2 - 10  # Move text up by 20 pixels

    for i, line in enumerate(text.split("\n")):
        line_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        line_x = bubble_x + (bubble_width - line_size[0]) // 2
        line_y = text_y + i * (line_size[1] + 10)
        cv2.putText(frame, line, (line_x, line_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('Data/haarcascade_frontalface_default.xml')
# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Path to your custom comic bubble image
bubble_image_path = r'Images/nah.png'

frame_count = 0  # Counter for skipping frames

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))

    # Detect landmarks and process each face
    if len(faces) > 0:
        _, landmarks_list = facemark.fit(frame, faces)
        for landmarks in landmarks_list:
            expression = detect_expression(landmarks[0])
            add_comic_bubble(frame, faces[0], expression, bubble_image_path)
        cartoon_frame = cartoon_ps2_style(frame)
    else:
        cartoon_frame = cartoon_ps2_style(frame)  # Apply cartoon effect even if no faces are detected

    cv2.imshow('Cartoon', cartoon_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()