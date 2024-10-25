# install opencv
import cv2

# Read an image from file
image = cv2.imread('c:\Z_Alien\Computer Vision\image.jpg')  # Replace 'input_image.jpg' with your image path
 
# # Show the original image
# cv2.imshow('Original Image', image)
 
# cv2.waitKey(0)  # Wait until any key is pressed
# cv2.destroyAllWindows()

# Cropping the image (here we're cropping a 100x100 square from the top-left corner)
cropped_image = image[0:100, 0:100]
 
# Show the cropped image
cv2.imshow('Cropped Image', cropped_image)
 
cv2.waitKey(0)
cv2.destroyAllWindows()

# Resizing the image (scaling it down to 50% of its original size)
resized_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5) 
# Setting (0, 0) for the size and using the fx=0.5 and fy=0.5 parameters scale the width and height by 50% effectively resizing the image to half its original size
 
# Show the resized image
cv2.imshow('Resized Image', resized_image)