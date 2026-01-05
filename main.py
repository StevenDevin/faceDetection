import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import os
import time
from datetime import datetime

# ===============================
# 1. FUNGSI LOAD GAMBAR (FIX)
# ===============================
def load_face_image(path):
    img = cv2.imread(path)              # BGR
    if img is None:
        raise ValueError("Gambar gagal dibaca")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(img, dtype=np.uint8)

# ===============================
# 2. MEDIAPIPE SETUP
# ===============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

EAR_THRESHOLD = 0.10
MAR_THRESHOLD = 0.35

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LIPS = [61, 291, 39, 181, 17, 405]

# ===============================
# 3. LOAD DATABASE WAJAH
# ===============================
known_face_encodings = []
known_face_names = []
dataset_path = "dataset_wajah"

print("Sedang memuat database wajah...")

for filename in sorted(os.listdir(dataset_path)):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(dataset_path, filename)
        try:
            image = load_face_image(path)
            print(path, image.dtype, image.shape)

            enc = face_recognition.face_encodings(image, model="small")
            if not enc:
                print(f" -> [SKIP] Tidak ada wajah")
                continue

            known_face_encodings.append(enc[0])
            name = filename.split("_")[0].capitalize()
            known_face_names.append(name)
            print(f" -> [OK] {name}")

        except Exception as e:
            print(f" -> [ERROR] {filename}: {e}")

print(f"Total wajah tersimpan: {len(known_face_encodings)}")

# ===============================
# 4. FUNGSI BANTU
# ===============================
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_ear(lm, idx, w, h):
    pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in idx]
    v1 = distance(pts[1], pts[5])
    v2 = distance(pts[2], pts[4])
    h_ = distance(pts[0], pts[3])
    return (v1 + v2) / (2*h_) if h_ > 0 else 0

def get_mar(lm, idx, w, h):
    pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in idx]
    h_ = distance(pts[0], pts[1])
    return distance(pts[4], pts[5]) / h_ if h_ > 0 else 0

# ===============================
# 5. PROGRAM UTAMA
# ===============================
cap = cv2.VideoCapture(0)
time.sleep(2)

blink = smile = verified = False
detected_name = ""

print("Kamera siap. Tekan Q untuk keluar.")

# Counter stabilisasi
blink_detected = False
mouth_detected = False
verified = False

blink_counter = 0
mouth_counter = 0

BLINK_FRAMES = 3
MOUTH_FRAMES = 5

def reset_verification():
    global blink_detected, mouth_detected, verified
    global blink_counter, mouth_counter, detected_name

    blink_detected = False
    mouth_detected = False
    verified = False

    blink_counter = 0
    mouth_counter = 0
    detected_name = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)

    h, w, _ = frame.shape
    result = face_mesh.process(rgb)

    box_color = (0,165,255)

    if not blink_detected:
        status_text = "Silakan KEDIP"
    elif not mouth_detected:
        status_text = "Silakan BUKA MULUT"
    else:
        status_text = "Verifikasi Wajah..."

    if result.multi_face_landmarks:
        for face in result.multi_face_landmarks:
            lm = face.landmark
            ear = (get_ear(lm, LEFT_EYE, w, h) + get_ear(lm, RIGHT_EYE, w, h)) / 2
            mar = get_mar(lm, LIPS, w, h)

            # ===============================
            # 1️⃣ DETEKSI KEDIP (WAJIB DULU)
            # ===============================
            if not blink_detected:
                if ear < EAR_THRESHOLD:
                    blink_counter += 1
                else:
                    blink_counter = 0

                if blink_counter >= BLINK_FRAMES:
                    blink_detected = True
                    print("✅ Kedip terdeteksi")

            # ===============================
            # 2️⃣ DETEKSI BUKA MULUT (SETELAH KEDIP)
            # ===============================
            elif not mouth_detected:
                if mar > MAR_THRESHOLD:
                    mouth_counter += 1
                else:
                    mouth_counter = 0

                if mouth_counter >= MOUTH_FRAMES:
                    mouth_detected = True
                    print("✅ Buka mulut terdeteksi")

            # ===============================
            # 3️⃣ FACE RECOGNITION
            # ===============================
            elif not verified and known_face_encodings:
                small = cv2.resize(rgb, (0,0), fx=0.25, fy=0.25)
                small = np.ascontiguousarray(small)

                locs = face_recognition.face_locations(small, model="hog")
                encs = face_recognition.face_encodings(small, locs)

                for enc in encs:
                    matches = face_recognition.compare_faces(
                        known_face_encodings, enc, tolerance=0.5
                    )
                    if True in matches:
                        idx = matches.index(True)
                        detected_name = known_face_names[idx]
                        verified = True
                        print(f"✅ ABSEN: {detected_name} | {datetime.now()}")

            # ===============================
            # DEBUG DISPLAY
            # ===============================
            cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, f"MAR: {mar:.3f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    if verified:
        status_text = f"Halo, {detected_name}"
        box_color = (0,255,0)

    cv2.rectangle(frame, (0,h-50), (w,h), box_color, -1)
    cv2.putText(frame, status_text, (20,h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Sistem Absensi", frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        print("Masukkan password untuk reset:")
        password = input()
        if password == "admin123":
            reset_verification()
            print(" Sistem di-reset")

cap.release()
cv2.destroyAllWindows()
