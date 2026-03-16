import cv2

img = cv2.imread("../data/video2/frames/000000.png")

x1,y1,x2,y2 = 5,400,1915,500

cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),3)

cv2.imwrite("test_box.png",img)

print("saved test_box.png")