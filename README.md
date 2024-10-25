# Computer-Vision-Project

## Project 1: Basic Image Manipulation with OpenCV

In this section, we explore basic image manipulation techniques using OpenCV. This project allows us to understand fundamental operations such as adding text, drawing shapes, and applying transformations (rotation, scaling, and translation) to images. These skills form the foundation for more advanced computer vision tasks.

Step 1: Adding Text to an Image
The first step involves reading an image and adding text to it. We use the cv2.putText() function to overlay text on the image.

python
Copy code# Read the image
image = cv2.imread('example.jpg')  # Replace 'example.jpg' with your image path
 
# Add text to the image
cv2.putText(image, 'OpenCV Tutorial', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  
# Text: 'OpenCV Tutorial', position: (50, 50), font: Hershey Simplex, size: 1, color: white, thickness: 2
 
# Display the image with text
cv2.imshow('Image with Text', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.putText(): Adds text to the image.

Parameters: Text content, position, font, scale, color, and thickness.

Step 2: Drawing Shapes on an Image
Next, we explore how to draw basic shapes like circles, rectangles, and lines on an image. This is useful for highlighting specific regions or marking objects.

python
Copy code# Import OpenCV library
import cv2
 
# Read the image
image = cv2.imread('example.jpg')  # Replace 'example.jpg' with your image path
 
# Draw a circle on the image
cv2.circle(image, (150, 150), 50, (255, 0, 0), 2)  # Center at (150, 150), radius: 50, color: blue, thickness: 2
 
# Draw a rectangle on the image
cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), 3)  # Top-left (50, 50), bottom-right (200, 200), color: green, thickness: 3
 
# Draw a line on the image
cv2.line(image, (10, 10), (300, 300), (0, 0, 255), 1)  # From (10, 10) to (300, 300), color: red, thickness: 1
 
# Display the image with shapes
cv2.imshow('Image with Shapes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.circle(): Draws a circle at a specified position.

cv2.rectangle(): Draws a rectangle between two corners.

cv2.line(): Draws a line between two points.

Step 3: Applying Transformations (Rotation, Scaling, and Translation)
Now we apply transformations to manipulate the image, including rotating, scaling, and translating.

a. Rotating the Image
We rotate the image by 45 degrees around its center.

python
Copy codeimport cv2
import numpy as np
 
# Load the image
image = cv2.imread('example.jpg')  # Replace 'example.jpg' with your image path
 
# Get image dimensions
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
 
# Apply rotation
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)  # Rotate by 45 degrees
rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
 
# Display the rotated image
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.getRotationMatrix2D(): Creates a transformation matrix for rotating the image.

cv2.warpAffine(): Applies the rotation to the image.

b. Scaling the Image
We resize the image by scaling it 1.5 times in both width and height.

python
Copy code# Apply scaling
scaled_image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)  # Scale by 1.5x
 
# Display the scaled image
cv2.imshow('Scaled Image', scaled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.resize(): Resizes the image based on specified scaling factors (fx and fy).

c. Translating the Image
We shift the image 50 pixels along both the x and y axes.

python
Copy code# Apply translation
translation_matrix = np.float32([[1, 0, 50], [0, 1, 50]])  # Translate by 50 pixels in x and y directions
translated_image = cv2.warpAffine(image, translation_matrix, (w, h))
 
# Display the translated image
cv2.imshow('Translated Image', translated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.warpAffine(): Applies translation to shift the image.

## Conclusion
In this section, we covered the following basic image manipulation techniques:

Adding text to an image.

Drawing shapes like circles, rectangles, and lines.

Applying transformations such as rotation, scaling, and translation.

These basic operations form the groundwork for more advanced image processing tasks in OpenCV, which we will explore in future projects. Stay tuned for the next lecture, where we’ll dive into real-time face detection and continue our journey in computer vision!
