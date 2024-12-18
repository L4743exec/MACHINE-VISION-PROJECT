import cv2
import numpy as np

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
    K = 4  # Fewer colors for faster quantization
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

def add_comic_bubble_with_image(frame, faces, bubble_image_path, text="NAH\nI'D\nWIN", 
                                text_color=(0, 0, 0), bubble_size_factor=1.5, 
                                bubble_offset=(80, -10), text_offset=(0, 0), font_thickness=1):
    bubble_image = cv2.imread(bubble_image_path, cv2.IMREAD_UNCHANGED)  # Load bubble image with alpha channel

    for (x, y, w, h) in faces:
        # Resize the bubble image relative to the face size
        bubble_width = int(w * bubble_size_factor)  # Width scales with face width
        bubble_height = int(h * bubble_size_factor // 2)  # Height scales with face height
        resized_bubble = cv2.resize(bubble_image, (bubble_width, bubble_height), interpolation=cv2.INTER_AREA)

        # Calculate position for overlaying the bubble
        bubble_x = x + w // 2 - bubble_width // 2 + bubble_offset[0]  # Horizontal offset
        bubble_y = y - bubble_height + bubble_offset[1]  # Vertical offset (e.g., above the face)

        # Ensure the bubble does not go out of the frame bounds
        bubble_x = max(0, min(bubble_x, frame.shape[1] - bubble_width))
        bubble_y = max(0, min(bubble_y, frame.shape[0] - bubble_height))

        # Overlay the bubble image with transparency
        for c in range(3):  # Iterate over color channels
            alpha = resized_bubble[:, :, 3] / 255.0  # Extract alpha channel
            frame[bubble_y:bubble_y + bubble_height, bubble_x:bubble_x + bubble_width, c] = \
                (alpha * resized_bubble[:, :, c] + 
                 (1 - alpha) * frame[bubble_y:bubble_y + bubble_height, bubble_x:bubble_x + bubble_width, c])

        # Add multi-line text inside the bubble
        lines = text.split("\n")  # Split text into lines
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        line_spacing = 10  # Spacing between lines

        # Calculate text position relative to the bubble
        text_size = [cv2.getTextSize(line, font, font_scale, font_thickness)[0] for line in lines]
        total_height = sum(size[1] for size in text_size) + (len(lines) - 1) * line_spacing

        # Start drawing text at the center of the bubble with additional offsets
        start_y = bubble_y + bubble_height // 2 - total_height // 2 + text_offset[1]

        for i, line in enumerate(lines):
            text_x = bubble_x + bubble_width // 2 - text_size[i][0] // 2 + text_offset[0]  # Center horizontally
            text_y = start_y + i * (text_size[i][1] + line_spacing) + text_size[i][1] // 2  # Adjust for line spacing
            cv2.putText(frame, line, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(r'Data/haarcascade_frontalface_default.xml')

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
        print("Error: Could not read frame.")
        break

    # Resize input frame for better performance
    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    if frame_count % 2 == 0:  # Skip every 2nd frame to improve performance
        continue

    # Detect faces
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=4, minSize=(40, 40))

    # Apply cartoon PS2-style effect (assuming this function is defined elsewhere)
    cartoon_frame = cartoon_ps2_style(frame)

    # Add comic text bubbles to the cartoon frame
    add_comic_bubble_with_image(cartoon_frame, faces, bubble_image_path, text="NAH\nI'D\nWIN")

    # Display the original and cartoon-style frames
    cv2.imshow('Original Video', frame)
    cv2.imshow('Cartoon PS2 Style with Comic Bubble', cartoon_frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()