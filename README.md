# Computer-Vision-Project

## Project 1: Image Manipulation with OpenCV

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


##################################################################

Section Recap- Project-2
### **Project 2: Real-Time Face Detection with OpenCV**



In this project, we take our computer vision journey to the next level by building a real-time face detection system using your PC's camera. The aim of this project is to access the system's webcam and detect faces in real-time using a pre-trained Haar cascade classifier.



#### **Step 1: Importing Libraries**



We will begin by importing the necessary libraries, **OpenCV** and **NumPy**. OpenCV is a powerful library for image processing, while NumPy is used for handling arrays and matrices of data.



```python

import cv2

import numpy as np

```



#### **Step 2: Loading the Haar Cascade Classifier**



The Haar cascade classifier is a pre-trained model that can detect faces. OpenCV provides us with this model in the form of an XML file. This file contains the data needed to identify faces in an image.



We will load this file into our program using the `cv2.CascadeClassifier()` function and reference the pre-trained face detection model stored in OpenCV’s data.



```python

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

```



#### **Step 3: Accessing the Webcam**



To capture live video, we will use the `cv2.VideoCapture()` function. The parameter `0` is used to select the default webcam. If you have multiple cameras connected, you can change the number to use a different camera.



```python

video_capture = cv2.VideoCapture(0)



if not video_capture.isOpened():

    print("Error accessing the camera")

    exit()

```



#### **Step 4: Capturing Frames in a Loop**



We’ll create a while loop to continuously capture frames from the webcam. Each frame will be processed one by one to detect faces in real time.



```python

while True:

    # Capture frame-by-frame

    ret, frame = video_capture.read()

   

    if not ret:

        print("Error reading frame from webcam")

        break

```



#### **Step 5: Converting Frames to Grayscale**



Face detection works more efficiently with grayscale images, as it reduces the amount of data and makes detection faster. We’ll convert each frame to grayscale using the `cv2.cvtColor()` function.



```python

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

```



#### **Step 6: Detecting Faces**



Now that we have the grayscale frame, we will detect faces using the `detectMultiScale()` function. This function finds faces of different sizes in the image. It takes several parameters:

- **Scale Factor**: Adjusts for faces of different sizes.

- **minNeighbors**: Sets the minimum number of neighboring rectangles required to retain a face.

- **minSize**: Specifies the minimum size of the detected face.



```python

    faces = face_cascade.detectMultiScale(

        gray,

        scaleFactor=1.1,

        minNeighbors=5,

        minSize=(30, 30)

    )

```



---



### **Project 2 Continued: Drawing Rectangles Around Detected Faces**



Now that we’ve successfully detected faces in the video feed, it’s time to enhance our program by visually marking these detections. We’ll draw rectangles around the detected faces, making it easier to identify them in the video frame.



#### **Step 7: Adding a For Loop to Process Detected Faces**



In this step, we will loop through the list of detected faces and draw a rectangle around each one. The `detectMultiScale()` function provides coordinates for the top-left and bottom-right corners of the face, which we’ll use to draw these rectangles.



```python

for (x, y, w, h) in faces:

    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

```

- `x, y`: Top-left corner of the rectangle.

- `w, h`: Width and height of the rectangle.

- `(255, 0, 0)`: Color of the rectangle (in this case, blue). You can modify the color to red or green.

- `2`: Thickness of the rectangle in pixels.



#### **Step 8: Displaying the Image with Detected Faces**



Once the rectangles are drawn, we need to display the video frame with the detection rectangles. For this, we use `cv2.imshow()` to show the video feed.



```python

cv2.imshow('Face Detection', frame)

```



#### **Step 9: Exiting the Loop**



We need to provide a way for users to exit the loop and stop the program. We’ll listen for the 'q' key, and when pressed, the program will break out of the loop and close all OpenCV windows.



```python

if cv2.waitKey(1) & 0xFF == ord('q'):

    break

```



#### **Step 10: Releasing Resources**



After the loop ends, it’s important to free up the webcam and close any windows opened by OpenCV.



```python

video_capture.release()

cv2.destroyAllWindows()

```



#### **The Full Program Code (with Comments)**



Here’s the full code, including comments to explain each step:



```python

# Import necessary libraries

import cv2

import numpy as np



# Load the pre-trained Haar cascade classifier for face detection

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



# Start capturing video from the webcam

video_capture = cv2.VideoCapture(0)



if not video_capture.isOpened():

    print("Error accessing the camera")

    exit()



# Main loop to process each video frame

while True:

    # Capture frame-by-frame

    ret, frame = video_capture.read()

   

    if not ret:

        print("Error reading frame from webcam")

        break



    # Convert the captured frame to grayscale

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    # Detect faces in the grayscale image

    faces = face_cascade.detectMultiScale(

        gray,

        scaleFactor=1.1,

        minNeighbors=5,

        minSize=(30, 30)

    )



    # Draw a rectangle around each detected face

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)



    # Display the resulting frame

    cv2.imshow('Face Detection', frame)



    # Break the loop if 'q' is pressed

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break



# Release the video capture object and close all windows

video_capture.release()

cv2.destroyAllWindows()

```



### **Face Reveal Demo**



After writing the code, the instructor demonstrates the program by showing the webcam capturing their face. The program detects their face and draws a blue rectangle around it. They also showcase how the program detects other objects, such as a Doraemon figure in the background.



The program is fully functional but may lag if a heavy recording software is running in the background.



### **Conclusion of Project 2**



With this, we have completed Project 2. You now have a basic real-time face detection system using OpenCV and Python, capable of identifying and marking faces in live video streams. In the next project, we’ll dive even deeper into computer vision. Stay tuned!




