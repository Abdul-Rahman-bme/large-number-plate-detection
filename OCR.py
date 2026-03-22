import cv2
# Replace with the URL shown in your app
url = "http://192.168.1.172:8080/video" 
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret: break
    cv2.imshow("Phone Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
