import cv2
import mediapipe as mp
import os

# ตั้งค่า MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# เปิดกล้อง
cap = cv2.VideoCapture(0)

# ฟังก์ชันสำหรับคำนวณระยะห่างระหว่างสองจุด
def calculate_distance(point1, point2):
    return abs(point1.x - point2.x)  # ระยะในแนวนอน (x-axis)

# ฟังก์ชันสำหรับปรับระดับเสียง (เฉพาะ macOS)
def set_volume_mac(level):
    """
    ปรับระดับเสียงของ macOS
    :param level: ระดับเสียงที่ต้องการ (0-100)
    """
    level = max(0, min(level, 100))  # จำกัดเสียงระหว่าง 0-100
    os.system(f"osascript -e 'set volume output volume {level}'")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # แปลงภาพจาก BGR เป็น RGB สำหรับการประมวลผลด้วย MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # แปลงกลับเป็น BGR เพื่อแสดงผล
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # ตรวจจับ Landmark ของมือ
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # วาด Landmark ลงบนภาพ
            mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # ตรวจจับระยะห่างในแนวนอน
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            horizontal_distance = calculate_distance(thumb_tip, index_tip)

            # แปลงระยะเป็นค่าระดับเสียง
            volume_level = int(horizontal_distance * 100)  # คูณ 100 เพื่อแปลงเป็นเปอร์เซ็นต์
            volume_level = max(0, min(volume_level, 100))  # จำกัดเสียงระหว่าง 0-100

            # ปรับระดับเสียง
            set_volume_mac(volume_level)

            # แสดงข้อมูลบนหน้าจอ
            cv2.putText(frame_bgr, f"Volume: {volume_level}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f"Volume: {volume_level}%")

    # แสดงผลภาพ
    cv2.imshow('Hand Tracking', frame_bgr)

    # กด Esc เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
