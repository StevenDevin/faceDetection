import cv2
import os

cap = cv2.VideoCapture(0)
save_dir = "dataset/steven"
os.makedirs(save_dir, exist_ok=True)

count = 0
while True:
    ret, frame = cap.read()
    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1)
    if key == ord('c'):
        cv2.imwrite(f"{save_dir}/{count}.jpg", frame)
        print("Saved", count)
        count += 1
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
