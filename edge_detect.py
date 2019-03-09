import cv2

img = cv2.imread(sys.argv[1])
edges = cv2.Canny(img, 100, 200)
cv2.imwrite(sys.argv[2], edges)
