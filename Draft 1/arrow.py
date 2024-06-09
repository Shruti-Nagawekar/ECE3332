import cv2
import numpy as np
import cv2
import numpy as np

# Function to detect arrows in a frame
def detect_arrow(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
        if len(approx) >= 7:  
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
            x = approx.ravel()[0]
            y = approx.ravel()[1] - 10
            cv2.putText(frame, "Arrow", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    return frame

# Capture video from file
cap = cv2.VideoCapture('video.mp4') 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_with_arrows = detect_arrow(frame)
    
    cv2.imshow('Arrow Detection', frame_with_arrows)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
