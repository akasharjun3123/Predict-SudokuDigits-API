import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import base64
from PIL import Image
import io

# FUNCTION 1. THERSHOLDING THE IMAGE
def preProcess(img):
    greyImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   # CONVERTING IMAGE TO GREY SCALE
    blurredImg = cv2.GaussianBlur(greyImg,(5,5),0)     # APPLYING GAUSSIAN BLURR ??
    thresholdImg = cv2.adaptiveThreshold(blurredImg, 255, 1, 1, 15, 4)  # APPLYING THERSHOLDING TO THE BLURRED IMAGE
    return thresholdImg

# FUNCTION 2. FINDING THE BIGGEST CONTOUR
def findBiggestContour(contours):
    sudokuContour = np.array([])
    maxArea = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area>500:
            peri = cv2.arcLength(i,closed=True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)  
            #print("Approx",approx)      # FINDING THE BIGEST CONTOUR
            if area>maxArea and len(approx)== 4:
                maxArea = area
                sudokuContour = approx
    return sudokuContour


# FUNCTION 3. REORDERING THE POINTS
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

# FUNCTION 4. SPLITTING THE BOXES IN SUDOKU GRID
def getDigitBoxes(img):
    rows = np.vsplit(img,9)
    boxes = []
    for row in rows:
        cols = np.hsplit(row,9)
        for box in cols:
            boxes.append(box)
    return boxes

# FUNCTION 5. PREPROCESSING AND RESHAPING THE DIGIT BOXES
def reshapeDigits(digits,border):
    resized_digits = []
    for dig in digits:
        cropped_image = dig[border:-border, border:-border]
        cropped_image = cv2.resize(cropped_image,(28,28))
        resized_digits.append(cropped_image)

    resized_digits = np.array(resized_digits)
    return  resized_digits

def preProcessDigits(digits):
    newDigits = []
    for dig in digits:
        dig = preProcess(dig)
        dig = dig/255
        newDigits.append(dig)
    newDigits=np.array(newDigits)
    return newDigits


# FUNCTION 6. PRINT THE SUDOKU
def printSudoku(arr):
    count =0
    for i in range(9):
        for j in range(9):
            print(arr[count], end = " ")
            count=count+1
            
        print()

def showStackedImages(imageArr):
    stacked_images = np.hstack((imageArr))
    cv2.imshow("img", stacked_images)
    cv2.waitKey(0)

def showDigitBoxes(boxes):
    plt.figure(figsize=(6, 6))
    for i in range(81):
        plt.subplot(9, 9, i+1)
        plt.imshow(boxes[i])
        plt.axis('off')
    plt.show()

def image_to_json(image_data):
    # Convert NumPy array to bytes
    img_bytes = image_data.tobytes()
    # Encode the image data as Base64
    base64_image = base64.b64encode(img_bytes).decode('utf-8')
    # Create a JSON object
    json_data = {
        "image": base64_image
    }
    return json.dumps(json_data)
    
def json_to_image(json_data, image_path):
    # Load JSON data
    data = json.loads(json_data)
    # Decode Base64 image data
    base64_image = data["image"]
    image_data = base64.b64decode(base64_image)
    # Convert bytes to image
    image = Image.open(io.BytesIO(image_data))
    # Save the image
    image.save(image_path)

def json_to_image(json_data, HEIGHT, WIDTH, CHANNELS):
    # Load JSON data
    data = json.loads(json_data)
    # Decode Base64 image data
    base64_image = data["image"]
    img_bytes = base64.b64decode(base64_image)
    # Convert bytes to NumPy array
    image_data = np.frombuffer(img_bytes, dtype=np.uint8)
    # Reshape NumPy array to image shape
    image = image_data.reshape((HEIGHT, WIDTH, CHANNELS))  # Replace HEIGHT, WIDTH, CHANNELS with your image dimensions
    # Convert NumPy array to PIL Image
    image = Image.fromarray(image)
    # Save the image
    return image
        






