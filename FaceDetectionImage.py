import cv2

# read image file
img = cv2.imread('C:\\Users\\waghm\\Documents\\CV\\cr7.jpg',1)

# getting image features
type(img)
print(img)
img.shape


# preview the image
cv2.imshow("CR7",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Cascade Classifier Object
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# converting image to greyscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# look for face coordinates in the image
faces = face_cascade.detectMultiScale(img_gray, scaleFactor = 1.05, minNeighbors = 5)

# creating rectangle around the face
for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0),3)     # reshaped = cv2.resize(img, int(img.shape[1]/2),int(img.shape[0])))
    

# display image with face identified
cv2.imshow("Gray", img)
cv2.waitKey(0)
cv2.destroyAllWindows()