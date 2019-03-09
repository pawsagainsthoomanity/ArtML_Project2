import cv2, sys
import numpy as np

im_gray = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# cv2.imwrite("grayscale.png", im_bw)

print(sys.argv[1])


# edges = cv2.Canny(im_bw, 100, 200)
# cv2.imwrite(sys.argv[2], edges)
im_blur = cv2.GaussianBlur(im_bw,(15,15),0)

for i in range(10):
	im = cv2.filter2D(im_blur, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))
	# im_blur = cv2.blur(im_bw,(10,10))  

	(thresh_2, im_bw) = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)



cv2.imwrite("result.png", im_bw)
