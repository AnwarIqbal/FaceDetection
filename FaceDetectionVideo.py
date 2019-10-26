import cv2


# cascade classifier object
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# capture video from webcam
vid = cv2.VideoCapture(0)

# frame counter
a = 1

while True:
    a = a + 1
    check, frame = vid.read()
    # print(check)
    # print(frame)
    
    # convert video frame from RGB to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # look for face coordinates in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.03, minNeighbors = 5)
    
    # creating rectangle around the face
    for x,y,w,h in faces:
        live = cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0),3)
    
    cv2.imshow("Window", live)
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break

print(a) # print number of frames

vid.release()
cv2.destroyAllWindows()