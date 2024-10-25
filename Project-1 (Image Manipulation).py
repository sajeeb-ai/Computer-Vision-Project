## (Basic Image Manipulation)

## Drawing Shape

# Import OpenCV library
import cv2

# Read the image
# image = cv2.imread('image.jpg')  # Replace 'example.jpg' with your image path

# # Draw a circle on the image
# cv2.circle(image, (150, 150), 50, (255, 0, 0), 2)  # Center at (150, 150), radius 50, color blue, thickness 2

# # Draw a rectangle on the image
# cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), 3)  # Top-left (50, 50), bottom-right (200, 200), color green, thickness 3

# # Draw a line on the image
# cv2.line(image, (10, 10), (250, 250), (0, 0, 255), 1)  # From (10, 10) to (300, 300), color red, thickness 1

# # Display the image with shapes
# cv2.imshow('Image with Shapes', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

###################################################################
## Adding Text to Images
# image = cv2.imread("white.jpg")
# text = "Vision"
# position = (50,50)
# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 1
# color = (255,0,0)
# thickness = 2
# # Add text to the image
# cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)  
# # Text, position, font, scale, color, thickness

# resized_image = cv2.resize(image, (300,300))
# cv2.imshow("Text on Image", resized_image)
# # Display the image with text
# cv2.imshow('Image with Text', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##########################################################
import numpy as np

# Load the image
image = cv2.imread('image.jpg')  # Replace 'example.jpg' with your image path

# Get image dimensions
(h, w) = image.shape[:2]   # h = height, w = weight

# Apply rotation
center = (w // 2, h // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)  # Rotate by 45 degrees
rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

# Apply scaling
scaled_image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)  # Scale by 1.5x

# Apply translation
translation_matrix = np.float32([[1, 0, 50], [0, 1, 50]])  # Translate by 50 pixels in x and y directions
translated_image = cv2.warpAffine(image, translation_matrix, (w, h))

# Display the images with transformations
cv2.imshow('Rotated Image', rotated_image)
cv2.imshow('Scaled Image', scaled_image)
cv2.imshow('Translated Image', translated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


